## Amaç
Hazır ACNE04 üzerinde dört sınıflı şiddet sınıflandırması ve tek sınıflı lezyon tespiti.

## Veri ve bölünme
ACNE04, 1.457 yüz fotoğrafı ve 18.983 lezyon kutusu içerir. Kaynak Wu ve arkadaşları, ICCV 2019.
Şiddet Hayashi aralığıdır: 0 hafif (1–5), 1 orta (6–20), 2 şiddetli (21–50), 3 çok şiddetli (51 ve üzeri).
Sınıf sayıları: 513, 633, 182, 129.
Ölçüm makalenin fold 0 ayrımıdır: 1.165 eğitim tarafı, 292 test. Doğrulama yalnız 1.165’in içinden, tohum 42 ve katmanlı %15 ile ayrılmıştır.
Makaledeki test dökümü: hafif 103, orta 127, şiddetli 36, çok şiddetli 26.
(Not: Doğrulama kümesinin sınıf bazında tam sayıları data/split_ozet.csv dosyasından buraya aktarılacak.)

## Deney düzeni
İki sınıflandırıcı, ortak ayarlar, kazananın doğrulama macro-F1’i ile seçilmesi.

## Sınıflandırma sonuçları
(Tablo ve macro-F1 figürü buraya eklenecek)

## Artırma karşılaştırması
(Fark ve dört recall farkı buraya eklenecek)

## Hata örnekleri
(Kişi 1'in 10 görüntüsünden kısa okuma buraya eklenecek)

## Lezyon tespiti
(mAP@0.5 skoru, kutu sayısından üretilen şiddet skoru ve hiç kutu çıkmayan görüntü sayısı buraya eklenecek)

## Sınırlar
Tek fold, çok şiddetli sınıfta 26 test görüntüsü, klinik karar aracı değil.

## İlgili çalışmalar
Wu ve arkadaşları (ICCV 2019), ACNE04 üzerinde tek etiketli öğrenmede 5-fold ortalama %78,42 doğruluk bildirmiştir. Bu çalışmanın ölçümü fold 0 üzerindedir.