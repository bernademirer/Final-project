"""Ablasyon ozeti: her modelde tek degisikligin etkisi.

    python src/ablasyon_ozet.py

results/{model}_abl_{varyant}_eval.json dosyalarini (colab_ablasyon.ipynb uretir) ve
Asama 2 birlesik kosusunu (results.csv + {model}_asama2_val_eval.json) okur.
results/ablasyon_ozet.csv ve results/sekil_ablasyon.png yazar. Eksik kosu atlanir.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELLER = ["resnet50", "efficientnet_b0", "convnext_tiny", "densenet121"]
VARYANTLAR = {
    "temel": ("Temel (224, hafif artirma)", "lightgray"),
    "384": ("+384 piksel", "skyblue"),
    "guclu": ("+guclu artirma", "seagreen"),
    "denge": ("+dengeli ornekleme", "orchid"),
    "hepsi": ("Hepsi birlikte (Asama 2)", "darkorange"),
}


def abl_olcumu(model: str, varyant: str) -> dict | None:
    yol = RESULTS / f"{model}_abl_{varyant}_eval.json"
    return json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else None


def asama2_olcumu(model: str, csv: pd.DataFrame) -> dict | None:
    ad = f"{model}_asama2"
    val_yolu = RESULTS / f"{ad}_val_eval.json"
    if not val_yolu.exists() or ad not in csv.index or f"{ad}_tta" not in csv.index:
        return None

    def satir(isim: str) -> dict:
        r = csv.loc[isim]
        return {
            "dogruluk": float(r["dogruluk"]),
            "macro_f1": float(r["macro_f1"]),
            "recall": [float(r[f"recall_{i}"]) for i in range(4)],
        }

    val = json.loads(val_yolu.read_text(encoding="utf-8"))
    return {
        "val_macro_f1": val["val_macro_f1"],
        "val_macro_f1_tta": val["val_macro_f1_tta"],
        "test": satir(ad),
        "test_tta": satir(f"{ad}_tta"),
        "epoch": int(csv.loc[ad, "epoch"]),
    }


def ozet_tablosu() -> pd.DataFrame:
    csv_yolu = RESULTS / "results.csv"
    csv = pd.read_csv(csv_yolu).set_index("model") if csv_yolu.exists() else pd.DataFrame()
    satirlar = []
    for model in MODELLER:
        for varyant in VARYANTLAR:
            olcum = asama2_olcumu(model, csv) if varyant == "hepsi" else abl_olcumu(model, varyant)
            if olcum is None:
                continue
            test, test_tta = olcum["test"], olcum["test_tta"]
            satirlar.append(
                {
                    "model": model,
                    "varyant": varyant,
                    "val_f1": olcum["val_macro_f1"],
                    "val_f1_tta": olcum["val_macro_f1_tta"],
                    "test_f1": test["macro_f1"],
                    "test_f1_tta": test_tta["macro_f1"],
                    "test_dogruluk": test["dogruluk"],
                    **{f"recall_{i}": test["recall"][i] for i in range(4)},
                    "epoch": olcum.get("epoch"),
                }
            )
    df = pd.DataFrame(satirlar)
    if df.empty:
        return df
    temel = df[df.varyant == "temel"].set_index("model")
    df["test_fark"] = [
        row.test_f1 - temel.test_f1[row.model] if row.model in temel.index else None
        for row in df.itertuples()
    ]
    df["val_fark"] = [
        row.val_f1 - temel.val_f1[row.model] if row.model in temel.index else None
        for row in df.itertuples()
    ]
    return df


def ablasyon_figuru(df: pd.DataFrame) -> None:
    genislik = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, sutun, baslik in zip(axes, ["test_f1", "test_f1_tta"], ["TTA'siz", "TTA'li"]):
        for i, (varyant, (etiket, renk)) in enumerate(VARYANTLAR.items()):
            parca = df[df.varyant == varyant].set_index("model")
            xs, ys = [], []
            for j, model in enumerate(MODELLER):
                if model in parca.index:
                    xs.append(j + (i - 2) * genislik)
                    ys.append(parca.loc[model, sutun])
            if not xs:
                continue
            ax.bar(xs, ys, genislik, color=renk, label=etiket)
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.01, f"{y:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(range(len(MODELLER)))
        ax.set_xticklabels(MODELLER)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Test macro-F1 ({baslik})")
    axes[0].set_ylabel("Test macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle("Ablasyon: her degisiklik tek basina (fold 0, tek tohum)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(RESULTS / "sekil_ablasyon.png", dpi=120)
    plt.close(fig)


def main() -> None:
    df = ozet_tablosu()
    if df.empty:
        print("Hic ablasyon sonucu yok.")
        return
    df.to_csv(RESULTS / "ablasyon_ozet.csv", index=False, float_format="%.4f")
    ablasyon_figuru(df)
    eksik = [
        f"{m}/{v}" for m in MODELLER for v in VARYANTLAR
        if not ((df.model == m) & (df.varyant == v)).any()
    ]
    pd.set_option("display.width", 160)
    print(df[["model", "varyant", "val_f1", "val_f1_tta", "test_f1", "test_f1_tta", "test_fark", "recall_2", "recall_3"]].round(4).to_string(index=False))
    print("ablasyon_ozet.csv ve sekil_ablasyon.png yazildi.")
    if eksik:
        print("Eksik kosular:", ", ".join(eksik))


if __name__ == "__main__":
    main()
