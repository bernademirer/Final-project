import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. Macro-F1 Çubuğu Grafiği
try:
    # CSV dosyasını oku
    df = pd.read_csv('results/results.csv')
    
    # İstenen modelleri filtrele (yolov8n bu grafikte olmamalı)
    istenen_modeller = ['resnet50', 'efficientnet_b0', 'kazanan_aug_off', 'yolov8n_sayimdan_siddet']
    df_filtered = df[df['model'].isin(istenen_modeller)].copy()

    # Sayı olmayan (boş) değerleri 0 kabul et ki kod çökmesin
    df_filtered['macro_f1'] = pd.to_numeric(df_filtered['macro_f1'], errors='coerce').fillna(0)

    # Çubuk grafiği çizimi
    plt.figure(figsize=(10, 6))
    renkler = ['skyblue', 'lightgreen', 'salmon', 'orchid']
    bars = plt.bar(df_filtered['model'], df_filtered['macro_f1'], color=renkler[:len(df_filtered)])
    
    plt.ylim(0, 1.1)
    plt.ylabel('Macro-F1 Skoru')
    plt.title('Modellerin Macro-F1 Karşılaştırması')
    plt.xticks(rotation=15)

    # Barların üzerine değerlerini yazdır
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/sekil_macro_f1.png')
    plt.close()
    print("sekil_macro_f1.png basariyla olusturuldu (Bos/Sifir verilerle).")

except Exception as e:
    print(f"Macro-F1 grafikleri cizilirken hata: {e}")

# 2. Karışıklık Matrislerini Yan Yana Basma
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Kazanan model txt'si gelmediyse varsayılan olarak resnet50 arayalım
    kazanan_model = "resnet50"
    if os.path.exists('results/kazanan.txt'):
        with open('results/kazanan.txt', 'r') as f:
            kazanan_model = f.read().strip()

    artirmali_yol = f'results/cm_{kazanan_model}.png'
    artirmasiz_yol = f'results/cm_{kazanan_model}_aug_off.png'

    # Sol panel: Artırmalı matris
    if os.path.exists(artirmali_yol):
        img1 = Image.open(artirmali_yol)
        axes[0].imshow(img1)
        axes[0].set_title('Artırmalı Matris (Kazanan)')
    else:
        axes[0].text(0.5, 0.5, f'{artirmali_yol}\nbekleniyor...', ha='center', va='center')
    axes[0].axis('off')

    # Sağ panel: Artırmasız matris
    if os.path.exists(artirmasiz_yol):
        img2 = Image.open(artirmasiz_yol)
        axes[1].imshow(img2)
        axes[1].set_title('Artırmasız Matris (aug_off)')
    else:
        axes[1].text(0.5, 0.5, f'{artirmasiz_yol}\nbekleniyor...', ha='center', va='center')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('results/sekil_karisiklik.png')
    plt.close()
    print("sekil_karisiklik.png basariyla olusturuldu (Goruntuler gelene kadar yer tutucularla).")

except Exception as e:
    print(f"Karisiklik matrisi figurleri birlestirilirken hata: {e}")