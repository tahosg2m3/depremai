import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch_directml
import time

# 1. DONANIMI HAZIRLA
device = torch_directml.device()
print("--- KULLANILAN DONANIM: AMD GPU (DirectML) ---")

# 2. ONNX DOSTU DERİN MİMARİ
class DeepEarthquakePredictor(nn.Module):
    def __init__(self):
        super(DeepEarthquakePredictor, self).__init__()
        # Conv1: 3000 -> 997
        self.conv1 = nn.Conv1d(3, 32, kernel_size=10, stride=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # 997 -> 498
        
        # Conv2: 498 -> 163
        self.conv2 = nn.Conv1d(32, 64, kernel_size=10, stride=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 163 -> 81
        
        # Conv3: 81 -> 39
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.flatten = nn.Flatten()
        
        # Matematiksel hesaplama: 128 kanal * 39 uzunluk = 4992
        self.fc1 = nn.Linear(128 * 39, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 3. VERİ VE NORMALİZASYON
print("1. 4515 Deprem Yükleniyor...")
veri = torch.load('deprem_veri_seti.pt', weights_only=True)
X, y = veri['X'], veri['y']

X_mean, X_std = X.mean(dim=(1, 2), keepdim=True), X.std(dim=(1, 2), keepdim=True)
X = (X - X_mean) / (X_std + 1e-7)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 4. EĞİTİM
model = DeepEarthquakePredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

print("2. Yapay Zeka Eğitimi (100 Epoch)...")
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{100}] bitti.")

# 5. ONNX DIŞA AKTARIM (HATA ÇÖZÜLDÜ)
print("\n3. Model ONNX formatına dönüştürülüyor...")
model.eval()
model.to('cpu')
dummy_input = torch.randn(1, 3000, 3)

# Opset 11 ve sabit katmanlar sayesinde artık hata vermeyecek
torch.onnx.export(model, dummy_input, "deprem_beyni.onnx",
                  export_params=True, opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'], output_names=['output'])

print("İŞLEM TAMAM! Dosyalar hazır.")