Soru: Aşama 2'deki üç değişikliğin (384 piksel, güçlü artırma, sınıf dengeli örnekleme) her biri tek başına test macro-F1'ini nasıl etkiliyor?

Bu bir ek deneydir; Aşama 1 ve Aşama 2 test skorları görüldükten sonra tasarlanmıştır. Amaç model seçmek değil, değişikliklerin etkisini ayırmaktır.

Modeller: ResNet50, EfficientNet-B0, ConvNeXt-Tiny, DenseNet121.

Her model için dört koşu; her biri temelden yalnız bir noktada ayrılır:
- temel: 224 piksel, hafif artırma (yatay çevirme, en fazla 10° döndürme), normal örnekleme.
- 384: temel + girdi boyutu 384.
- guclu: temel + güçlü artırma (en az %85'i koruyan kırpma, parlaklık ve kontrast ±%20, doygunluk ±%10).
- denge: temel + sınıf dengeli örnekleme.

"Hepsi birlikte" karşılaştırması Aşama 2 koşusudur. TTA (yatay çevirme ortalaması) her koşuda ayrıca ölçülür.

Bütün koşular karma hassasiyetle (AMP) çalışır; temel koşu da bu yüzden Aşama 1'den ayrı olarak yeniden yapılır. Diğer ayarlar configs/ortak.yaml dosyasındaki gibidir: AdamW, öğrenme oranı 1e-4, ağırlık azaltma 0,01, batch 16, en fazla 30 epoch, sabır 5, tohum 42.

Okuma kuralı: tek fold ve tek tohum vardır. Doğrulama kümesi 175 görüntüdür ve Aşama 1 ile 2'de doğrulama sıralaması testle uyuşmamıştır. Bu yüzden 0,03'ten küçük macro-F1 farkları gürültü kabul edilir; bir değişiklik ancak modellerin çoğunda aynı yönde etki gösterirse yorumlanır.
