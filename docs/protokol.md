Soru: aynı modelde yatay çevirme ve en fazla 10° döndürme, test macro-F1’ini artırıyor mu?

Model, results/kazanan.txt içindeki addır. Seçim doğrulama macro-F1’i iledir.

Artırmalı koşu Kişi 1 veya Kişi 2 tarafından zaten yapılmıştır. Bu protokol yalnız artırmasız ikinci koşuyu ekler.

Sabitler: 224×224, ImageNet normalizasyonu, ImageNet ağırlığı, AdamW, öğrenme oranı 1e-4, ağırlık azaltma (weight decay) 0,01, batch 16, en fazla 30 epoch, doğrulama macro-F1’i 5 epoch iyileşmezse dur, tohum 42, çapraz entropi, sınıf ağırlığı yok. Değerler configs/ortak.yaml dosyasından okunur.

İki koşu arasındaki tek fark eğitim kümesindeki artırmadır (--aug on / --aug off). Doğrulama ve test görüntüleri iki koşuda da yalnız 224×224 ölçekleme ve ImageNet normalizasyonundan geçer.

Başarı ölçütü macro-F1’dir. Doğruluk tek başına karar ölçütü değildir; orta sınıf 633 görüntüyle doğruluğu yukarı çeker.

Test kümesi koşu bitene kadar yeni bir karar için kullanılmaz. Test skoru görüldükten sonra artırma, epoch veya öğrenme oranı değişmez.
