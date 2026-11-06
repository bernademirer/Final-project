from pathlib import Path
import pandas as pd
import shutil

# splits.csv dosyasını oku ve eğitim kümesinden seç
splits_df = pd.read_csv("data/splits.csv")
train_subset = splits_df[splits_df["kume"] == "train"]

ornek_dir = Path("data/ornekler")
ornek_dir.mkdir(parents=True, exist_ok=True)

etiketler_path = ornek_dir / "etiketler.txt"
with open(etiketler_path, "w", encoding="utf-8") as f:
    for lvl in [0, 1, 2, 3]:
        # Her level'dan 2'şer görüntü seç
        secilenler = train_subset[train_subset["level"] == lvl].head(2)
        for _, row in secilenler.iterrows():
            src = Path("data/images") / row["file_name"]
            dest = ornek_dir / row["file_name"]
            if src.exists():
                shutil.copy(src, dest)
            f.write(f"{row['file_name']} -> level: {lvl}\n")

print("Örnek görüntüler data/ornekler/ altına kopyalandı ve etiketler.txt oluşturuldu!")