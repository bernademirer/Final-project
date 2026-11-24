Soru: daha güçlü veri artırma, daha yüksek çözünürlük ve sınıf dengeli örnekleme, dört farklı mimaride test macro-F1'ini artırıyor mu?

Bu bir ek deneydir. Aşama 1 (docs/protokol.md) test skorları görüldükten sonra tasarlanmıştır; Aşama 1'in sonuçlarının yerine geçmez, ayrı raporlanır.

Modeller: ResNet50, EfficientNet-B0, ConvNeXt-Tiny, DenseNet121. Hepsi ImageNet ağırlığıyla başlar.

Aşama 1'e göre değişenler (hepsi birlikte, tek koşu):
- Girdi boyutu 224 yerine 384.
- Artırma (--aug guclu): görüntünün en az %85'ini koruyan rastgele kırpma, yatay çevirme, en fazla 10° döndürme, parlaklık ve kontrast ±%20, doygunluk ±%10. Renk tonu değişmez.
- Sınıf dengeli örnekleme: her epoch'ta sınıflar eşit olasılıkla çekilir (yerine koyarak).
- Karma hassasiyet (AMP), yalnız hız ve bellek için.

Sabit kalanlar: splits.csv, AdamW, öğrenme oranı 1e-4, ağırlık azaltma 0,01, batch 16, en fazla 30 epoch, sabır 5, tohum 42, çapraz entropi. Değerler configs/ortak.yaml dosyasından okunur.

TTA: test sırasında görüntü ve yatay çevrilmiş hâli için olasılıkların ortalaması. Yeniden eğitim gerektirmez.

Seçim: Aşama 2 kazananı doğrulama macro-F1'i en yüksek modeldir. TTA, doğrulamada daha iyiyse eklenir. Test her yapılandırma için bir kez ölçülür; hiçbir karar test skoruna göre verilmez.

Değişiklikler birlikte uygulandığından iyileşme tek tek bileşenlere ayrılamaz.
