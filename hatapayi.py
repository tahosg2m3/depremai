import onnxruntime as ort
import torch
import numpy as np
import time

print("--- 4515 DEPREMLİK DEVASA JÜRİ TESTİ BAŞLIYOR ---")
try:
    session = ort.InferenceSession("deprem_beyni.onnx")
    input_name = session.get_inputs()[0].name
except Exception as e:
    print("Hata: deprem_beyni.onnx dosyası bulunamadı!", e)
    exit()

veri = torch.load('deprem_veri_seti.pt', weights_only=True)
X_test = veri['X'].numpy()
y_test = veri['y'].numpy()

toplam = len(X_test)
mutlak_hata_toplami = 0
basarili_tahmin_sayisi = 0 

# Alarm Sayaçları
yanlis_alarm = 0           
kacirilan_deprem = 0       

print(f"Toplam {toplam} deprem tek tek analiz ediliyor...\n")
baslangic_zamani = time.time()

for i in range(toplam):
    sanal_sensor_verisi = X_test[i:i+1]
    gercek_siddet = y_test[i][0]

    ortalama = np.mean(sanal_sensor_verisi)
    sapma = np.std(sanal_sensor_verisi)
    sanal_sensor_verisi = (sanal_sensor_verisi - ortalama) / (sapma + 1e-7)
    sanal_sensor_verisi = sanal_sensor_verisi.astype(np.float32)

    tahmin_sonucu = session.run(None, {input_name: sanal_sensor_verisi})
    tahmin = tahmin_sonucu[0][0][0]

    hata_miktari = abs(gercek_siddet - tahmin)
    mutlak_hata_toplami += hata_miktari

    if hata_miktari <= 0.5:
        basarili_tahmin_sayisi += 1

    # --- MÜHENDİSLİK GÜVENLİĞİ (SAFETY MARGIN) ---
    # Gerçek tehlike sınırı 4.5
    # Yapay zeka alarm eşiği 4.1 (0.4 puanlık güvenlik payı)
    if gercek_siddet < 4.5 and tahmin >= 4.1:
        yanlis_alarm += 1
    elif gercek_siddet >= 4.5 and tahmin < 4.1:
        kacirilan_deprem += 1

bitis_zamani = time.time()

ortalama_sapma = mutlak_hata_toplami / toplam
basari_yuzdesi = (basarili_tahmin_sayisi / toplam) * 100

print("\n" + "="*50)
print("         🏆 TEKNOFEST JÜRİ RAPORU (GÜVENLİK PAYLI) 🏆")
print("="*50)
print(f"Test Edilen Toplam Deprem : {toplam}")
print(f"Toplam Test Süresi        : {bitis_zamani - baslangic_zamani:.2f} saniye\n")

print(f"📊 Ortalama Şiddet Sapması : ±{ortalama_sapma:.2f}")
print(f"🎯 ±0.5 İsabet Oranı       : %{basari_yuzdesi:.1f} ({toplam} sarsıntının {basarili_tahmin_sayisi} tanesi tam isabet)")
print(f"🚨 Yanlış Alarm Sayısı     : {yanlis_alarm} (Zararsız dalgaya alarm)")
print(f"⚠️ Kaçırılan Deprem Sayısı : {kacirilan_deprem} (Kritik tehlike atlaması)")
print("="*50)