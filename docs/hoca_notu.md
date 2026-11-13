## Amaç
Hazır ACNE04 üzerinde dört sınıflı şiddet sınıflandırması ve tek sınıflı lezyon tespiti.

## Veri ve bölünme
ACNE04, 1.457 yüz fotoğrafı ve 18.983 lezyon kutusu içerir. Kaynak Wu ve arkadaşları, ICCV 2019.
Şiddet Hayashi aralığıdır: 0 hafif (1–5), 1 orta (6–20), 2 şiddetli (21–50), 3 çok şiddetli (51 ve üzeri).
Sınıf sayıları: 513, 633, 182, 129.
Ölçüm makalenin fold 0 ayrımıdır: 1.165 eğitim tarafı, 292 test. Doğrulama yalnız 1.165’in içinden, tohum 42 ve katmanlı %15 ile ayrılmıştır.

| Küme | Hafif (0) | Orta (1) | Şiddetli (2) | Çok şiddetli (3) | Toplam |
|---|---|---|---|---|---|
| Eğitim | 348 | 430 | 124 | 88 | 990 |
| Doğrulama | 62 | 76 | 22 | 15 | 175 |
| Test | 103 | 127 | 36 | 26 | 292 |

Kaynak: data/split_ozet.csv. Test dökümü makaledeki fold 0 testiyle aynıdır.
Çok şiddetli sınıf testte yalnız 26 görüntüdür; bu sınıfın recall değeri tek bir görüntüyle yaklaşık 4 puan oynar ve oynak okunmalıdır.

## Deney düzeni
İki sınıflandırıcı, ortak ayarlar, kazananın doğrulama macro-F1’i ile seçilmesi.

## Sınıflandırma sonuçları
(Tablo ve macro-F1 figürü buraya eklenecek)

## Artırma karşılaştırması
(Fark ve dört recall farkı buraya eklenecek)

## Hata örnekleri
(Kişi 1'in 10 görüntüsünden kısa okuma buraya eklenecek)

## Lezyon tespiti
Tek sınıflı (lezyon) YOLOv8-Nano, 640 piksel girdi ve 50 epoch ile eğitilmiştir. Test kümesinde mAP@0.5 değeri 0,3118’dir.
Güven eşiği 0,25 üstündeki kutular sayılıp Hayashi aralığına konduğunda, bu sayımdan üretilen şiddet gerçek level ile karşılaştırılmıştır: doğruluk 0,3836, macro-F1 0,1712. Hafif sınıf recall’u 1,0000, orta 0,0709, şiddetli ve çok şiddetli 0,0000’dır; dedektör lezyonları eksik saydığı için tahminler hafif sınıfta toplanmıştır.
292 test görüntüsünün 71’inde eşiğin üstünde hiç kutu çıkmamış, bu görüntüler kural gereği hafif sayılmıştır.

## Sınırlar
Tek fold, çok şiddetli sınıfta 26 test görüntüsü, klinik karar aracı değil.

## İlgili çalışmalar
Wu ve arkadaşları (ICCV 2019), ACNE04 üzerinde tek etiketli öğrenmede 5-fold ortalama %78,42 doğruluk bildirmiştir. Bu çalışmanın ölçümü fold 0 üzerindedir.
