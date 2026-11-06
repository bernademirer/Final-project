from pathlib import Path
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Veri setini yükle
ds = load_dataset("Layered-Labs/ACNE04")
rows = ds["train"]

# 2. Görüntüleri diske kaydet (Faz 3)
out = Path("data/images")
out.mkdir(parents=True, exist_ok=True)

for row in rows:
    dest = out / row["file_name"]
    if not dest.exists():
        row["image"].save(dest)

print("Kaydedilen toplam dosya sayısı:", len(list(out.glob('*.jpg'))))

# 3. splits.csv dosyasını kilitle ve split_ozet.csv oluştur (Faz 4)
df = pd.DataFrame({
    "file_name": rows["file_name"],
    "level": rows["level"],
    "severity": rows["severity"],
    "num_lesions": rows["num_lesions_annotated"],
    "paper_split": rows["split"],
})

test = df[df.paper_split == "test"].copy()
trainval = df[df.paper_split == "train"].copy()

train, val = train_test_split(
    trainval, test_size=0.15, random_state=42, stratify=trainval["level"]
)

train["kume"] = "train"
val["kume"] = "val"
test["kume"] = "test"

splits = pd.concat([train, val, test], ignore_index=True)
splits = splits.drop(columns=["paper_split"])

Path("data").mkdir(parents=True, exist_ok=True)
splits.to_csv("data/splits.csv", index=False)

ozet = splits.groupby(["level", "kume"]).size().unstack(fill_value=0)
ozet.to_csv("data/split_ozet.csv")

print("splits.csv ve split_ozet.csv başarıyla oluşturuldu ve kilitlendi!")