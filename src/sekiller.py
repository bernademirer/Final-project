"""Hoca notunun iki figurunu uretir.

    python src/sekiller.py

results/results.csv okunur. Sayisi olmayan satir grafige girmez (0 diye cizilmez).
Karisiklik figurunde dosyasi olmayan panel bos kalir. Betik her durumda hatasiz biter.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Sira ve renk modele bagli; bir satir eksik olsa da digerlerinin rengi kaymaz.
# yolov8n bilerek yok: o satirdaki sayi mAP@0.5'tir, macro-F1 degildir.
MODELLER = {
    "resnet50": "skyblue",
    "efficientnet_b0": "lightgreen",
    "kazanan_aug_off": "salmon",
    "yolov8n_sayimdan_siddet": "orchid",
}


def macro_f1_figuru() -> None:
    csv_yolu = RESULTS / "results.csv"
    if not csv_yolu.exists():
        print("results.csv yok, sekil_macro_f1.png atlandi.")
        return
    df = pd.read_csv(csv_yolu)
    df = df[df["model"].isin(MODELLER)].copy()
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df = df.dropna(subset=["macro_f1"])
    df["sira"] = df["model"].map(list(MODELLER).index)
    df = df.sort_values("sira")

    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, "Henuz macro-F1 sayisi yok", ha="center", va="center", transform=ax.transAxes)
    else:
        bars = ax.bar(df["model"], df["macro_f1"], color=[MODELLER[m] for m in df["model"]])
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.02, f"{y:.4f}", ha="center", va="bottom")
    eksik = [m for m in MODELLER if m not in set(df["model"])]
    if eksik:
        ax.set_xlabel("Bekleniyor: " + ", ".join(eksik), fontsize=9, color="gray")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Modellerin macro-F1 karsilastirmasi (test, fold 0)")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "sekil_macro_f1.png", dpi=120)
    plt.close(fig)
    print(f"sekil_macro_f1.png yazildi. Cizilen: {list(df['model'])}. Eksik: {eksik}")


def karisiklik_figuru() -> None:
    kazanan_yolu = RESULTS / "kazanan.txt"
    kazanan = kazanan_yolu.read_text(encoding="utf-8").strip() if kazanan_yolu.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    paneller = [
        ("Artirmali (kazanan)", f"cm_{kazanan}.png" if kazanan else None),
        ("Artirmasiz (aug_off)", f"cm_{kazanan}_aug_off.png" if kazanan else None),
    ]
    for ax, (baslik, ad) in zip(axes, paneller):
        ax.axis("off")
        if ad and (RESULTS / ad).exists():
            ax.imshow(Image.open(RESULTS / ad))
            ax.set_title(f"{baslik}\n{ad}")
        else:
            mesaj = f"results/{ad}\nbekleniyor" if ad else "results/kazanan.txt\nbekleniyor"
            ax.text(0.5, 0.5, mesaj, ha="center", va="center")
            ax.set_title(baslik)
    fig.tight_layout()
    fig.savefig(RESULTS / "sekil_karisiklik.png", dpi=120)
    plt.close(fig)
    print(f"sekil_karisiklik.png yazildi. Kazanan: {kazanan or 'henuz yok'}")


if __name__ == "__main__":
    for adim in (macro_f1_figuru, karisiklik_figuru):
        try:
            adim()
        except Exception as hata:  # betik hata vermeden cikmali
            print(f"{adim.__name__} sirasinda hata: {hata}")
