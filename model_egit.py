import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch_directml

# DirectML tarafında bazı fused CPU yollarını tetiklememek için kapatıyoruz.
torch.backends.mkldnn.enabled = False

# 1) DONANIM
DEVICE = torch_directml.device()
print("╔══════════════════════════════════════════════════╗")
print("║   KULLANILAN DONANIM: AMD GPU (DirectML)        ║")
print("╚══════════════════════════════════════════════════╝")


class DirectMLConvRegressor(nn.Module):
    """DirectML ile uyumlu (LSTM içermeyen) deprem regresyon modeli."""

    def __init__(self, input_channels: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=9, stride=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.regressor(x)


# 2) VERİ
print("\n1. Veri seti yükleniyor...")
veri = torch.load("deprem_veri_seti.pt", map_location="cpu", weights_only=True)
X, y = veri["X"].float(), veri["y"].float()
print(f"   X: {X.shape}, y: {y.shape}")

# Özellik bazlı normalizasyon (kanal istatistikleri)
X_mean = X.mean(dim=(0, 1), keepdim=True)
X_std = X.std(dim=(0, 1), keepdim=True)
X = (X - X_mean) / (X_std + 1e-6)

dataset = TensorDataset(X, y)
train_size = int(len(dataset) * 0.85)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
print(f"   Eğitim: {len(train_dataset)} | Doğrulama: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3) MODEL
input_channels = X.shape[2]
model = DirectMLConvRegressor(input_channels=input_channels).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# 4) EĞİTİM
EPOCHS = 200
PATIENCE = 20
best_val_loss = float("inf")
patience_counter = 0
best_state = None

print("\n2. Eğitim başlıyor (200 epoch, erken durma aktif)...")
print("─" * 60)
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for bX, by in train_loader:
        bX = bX.to(DEVICE)
        by = by.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        pred = model(bX)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * bX.size(0)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for bX, by in val_loader:
            bX = bX.to(DEVICE)
            by = by.to(DEVICE)
            pred = model(bX)
            loss = criterion(pred, by)
            val_loss_sum += loss.item() * bX.size(0)

    train_loss = train_loss_sum / len(train_dataset)
    val_loss = val_loss_sum / len(val_dataset)

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nErken durma tetiklendi (patience={PATIENCE}).")
        break

if best_state is not None:
    model.load_state_dict(best_state)

print(f"\nEğitim süresi: {time.time() - start:.1f} saniye")
print(f"En iyi doğrulama kaybı: {best_val_loss:.6f}")

# 5) KAYIT
model_cpu = model.to("cpu").eval()
torch.save(
    {
        "model_state_dict": model_cpu.state_dict(),
        "input_channels": input_channels,
        "seq_len": int(X.shape[1]),
        "best_val_loss": best_val_loss,
    },
    "deprem_modeli_directml.pt",
)

print("\n3. Model kaydedildi: deprem_modeli_directml.pt")
print("İşlem tamam.")
