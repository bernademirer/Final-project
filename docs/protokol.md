Soru: aynı modelde yatay çevirme ve en fazla 10° döndürme, test macro-F1’ini artırıyor mu?

Model, results/kazanan.txt içindeki addır. Seçim doğrulama macro-F1’i iledir.

Sabitler: 224×224, ImageNet normalizasyonu, ImageNet ağırlığı, AdamW, öğrenme oranı 1e-4, batch 16, en fazla 30 epoch, doğrulama macro-F1’i 5 epoch iyileşmezse dur, tohum 42, çapraz entropi, sınıf ağırlığı yok.

Başarı ölçütü macro-F1’dir. Doğruluk tek başına karar ölçütü değildir; orta sınıf 633 görüntüyle doğruluğu yukarı çeker.

Test kümesi koşu bitene kadar yeni bir karar için kullanılmaz. Test skoru görüldükten sonra artırma, epoch veya öğrenme oranı değişmez.