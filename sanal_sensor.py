import onnxruntime as ort
import torch
import numpy as np
import time
import random

print("--- YAPAY ZEKA GÜVENİLİRLİK TESTİ BAŞLIYOR ---")

try:
    session = ort.InferenceSession("deprem_beyni.onnx")
    input_name = session.get_inputs()[0].name
except Exception as e:
    print("Hata: deprem_beyni.onnx dosyası bulunamadı!", e)
    exit()

veri = torch.load('deprem_veri_seti.pt', weights_only=True)
X_test = veri['X'].numpy()
y_test = veri['y'].numpy()

toplam_deprem = len(X_test)
print(f"Toplam {toplam_deprem} adet Kandilli verisi bulundu.\n")
print(f"{'Test':<5} | {'Gerçek Şiddet':<15} | {'Yapay Zeka Tahmini':<20} | {'Tepki Süresi':<15} | {'Sistem Kararı'}")
print("-" * 85)

test_edilecek_sayi = min(10, toplam_deprem)
test_indeksleri = random.sample(range(toplam_deprem), test_edilecek_sayi)

for i, idx in enumerate(test_indeksleri):
    sanal_sensor_verisi = X_test[idx:idx+1] 
    gercek_siddet = y_test[idx][0]

    # --- İŞTE EKSİK OLAN O HAYATİ DOKUNUŞ: NORMALİZASYON ---
    # Eğitirken yaptığımız gibi, modeli test ederken de veriyi ehlileştirmeliyiz!
    ortalama = np.mean(sanal_sensor_verisi)
    sapma = np.std(sanal_sensor_verisi)
    # Veriyi -1 ile +1 arasına sıkıştırıyoruz
    sanal_sensor_verisi = (sanal_sensor_verisi - ortalama) / (sapma + 1e-7)
    
    # ONNX kütüphanesi çok hassastır, rakam formatını ondalıklı float32'ye sabitliyoruz
    sanal_sensor_verisi = sanal_sensor_verisi.astype(np.float32) 
    # -----------------------------------------------------------

    baslangic_zaman = time.time()
    tahmin_sonucu = session.run(None, {input_name: sanal_sensor_verisi})
    yapay_zeka_tahmini = tahmin_sonucu[0][0][0]
    bitis_zaman = time.time()
    
    tepki_suresi = (bitis_zaman - baslangic_zaman) * 1000
    durum = "🚨 ALARM VERİLDİ" if yapay_zeka_tahmini >= 4.5 else "✅ GÜVENLİ (Pas Geçildi)"
    
    print(f"{i+1:<5} | {gercek_siddet:<15.1f} | {yapay_zeka_tahmini:<20.1f} | {tepki_suresi:<10.2f} ms  | {durum}")
    time.sleep(0.5) 

print("-" * 85)
print("TEST TAMAMLANDI!")