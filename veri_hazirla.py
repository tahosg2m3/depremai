import os
import zipfile
import obspy
import torch
import numpy as np
import re

ana_klasor = "." 
islenmis_veriler = []
etiketler = []
hedef_uzunluk = 3000 


print("\n--- 2. AŞAMA: VERİ MADENCİLİĞİ BAŞLADI ---")
for kok_dizin, alt_dizinler, dosyalar in os.walk(ana_klasor):
    
    # Hem .ko hem de .sac dosyalarını arıyoruz
    sismik_dosyalar = [d for d in dosyalar if d.lower().endswith((".ko", ".sac")) or ".ko" in d.lower() or ".sac" in d.lower()]
    
    if len(sismik_dosyalar) >= 3:
        try:
            # Şiddeti bulmak için sadece klasör ismine değil, tam yola bakıyoruz (Zipten çıkanlar için daha güvenli)
            tam_yol_klasor = os.path.abspath(kok_dizin)
            eslesme = re.search(r'M=([0-9.]+)', tam_yol_klasor)
            
            if not eslesme:
                continue 
            
            buyukluk = float(eslesme.group(1))
            
            eksen_verileri = []
            for dosya in sismik_dosyalar[:3]:
                tam_yol_dosya = os.path.join(kok_dizin, dosya)
                with open(tam_yol_dosya, "rb") as f:
                    st = obspy.read(f)
                    data = st[0].data
                    
                    if len(data) > hedef_uzunluk:
                        data = data[:hedef_uzunluk]
                    else:
                        data = np.pad(data, (0, hedef_uzunluk - len(data)), 'constant')
                        
                    eksen_verileri.append(data)
            
            birlesik_veri = np.stack(eksen_verileri, axis=1) 
            
            islenmis_veriler.append(birlesik_veri)
            etiketler.append(buyukluk)
            
            print(f"Başarılı -> Şiddet: {buyukluk} | Klasör: {os.path.basename(kok_dizin)[:30]}...")
            
        except Exception as e:
            pass

if islenmis_veriler:
    print("\n--- İŞLEM BİTTİ! PAKETLEME YAPILIYOR ---")
    X_tensor = torch.tensor(np.array(islenmis_veriler), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(etiketler), dtype=torch.float32).view(-1, 1)
    torch.save({'X': X_tensor, 'y': y_tensor}, 'deprem_veri_seti.pt')
    
    print(f"Harika! Toplam {len(islenmis_veriler)} adet deprem başarıyla işlendi.")
    print(f"Veri Boyutu (X): {X_tensor.shape}")
    print("Her şey 'deprem_veri_seti.pt' adlı tek bir dosyaya sıkıştırıldı!")
else:
    print("\nUygun klasör veya dosya bulunamadı.")