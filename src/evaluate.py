"""Test olcumu. results.csv yalniz --write-csv ile guncellenir.

    python src/evaluate.py --model efficientnet_b0 --weights results/efficientnet_b0_best.pt --split test
    python src/evaluate.py --model efficientnet_b0 --weights results/efficientnet_b0_best.pt --split test --write-csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ROOT, AcneSplit, get_model, get_transforms, load_cfg, read_splits  # noqa: E402

HEADER = [
    "model",
    "dogruluk",
    "macro_f1",
    "recall_0",
    "recall_1",
    "recall_2",
    "recall_3",
    "epoch",
    "sure_dk",
]
LABELS = [0, 1, 2, 3]


def upsert_csv(path: Path, values: list[str]) -> None:
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    if not rows:
        rows = [HEADER]
    model_index = rows[0].index("model")
    replaced = False
    for row in rows[1:]:
        if row and row[model_index] == values[0]:
            row[:] = values
            replaced = True
            break
    if not replaced:
        rows.append(values)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--csv", default="data/splits.csv")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--write-csv", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg()
    frame = read_splits(ROOT / args.csv)
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir
    dataset = AcneSplit(
        frame[frame.kume == args.split],
        image_dir,
        get_transforms(False, False, cfg),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)
    blob = torch.load(ROOT / args.weights, map_location=device, weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    recalls = recall_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    accuracy = float(accuracy_score(y_true, y_pred))
    score = float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    print(f"dogruluk {accuracy:.4f}")
    print(f"macro_f1 {score:.4f}")
    for index, value in enumerate(recalls):
        print(f"recall_{index} {float(value):.4f}")

    aug_off = "aug_off" in Path(args.weights).name
    cm_name = f"cm_{args.model}_aug_off.png" if aug_off else f"cm_{args.model}.png"
    fig, axis = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=axis,
    )
    axis.set_xlabel("tahmin")
    axis.set_ylabel("gercek")
    fig.tight_layout()
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / cm_name, dpi=120)
    plt.close(fig)
    print("matris", out / cm_name)

    if not args.write_csv:
        print("results.csv yazilmadi. Kalici satir icin --write-csv ekleyin.")
        return

    row_name = f"{args.model}_aug_off" if aug_off else args.model
    if row_name == f"{args.model}_aug_off":
        row_name = "kazanan_aug_off"
    epoch = blob.get("epoch", "") if isinstance(blob, dict) else ""
    minutes = ""
    if isinstance(blob, dict) and "minutes" in blob:
        minutes = f"{float(blob['minutes']):.2f}"
    upsert_csv(
        out / "results.csv",
        [
            row_name,
            f"{accuracy:.4f}",
            f"{score:.4f}",
            *[f"{float(value):.4f}" for value in recalls],
            str(epoch),
            minutes,
        ],
    )
    print("satir guncellendi", row_name)


if __name__ == "__main__":
    main()
