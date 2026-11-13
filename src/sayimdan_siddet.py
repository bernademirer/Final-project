"""Test kutularini Hayashi araligina cevirir. Guven esigi 0.25 sabittir.

mAP satiri bu esikle yeniden hesaplanmaz. 0 kutu hafif sayilir ve ayri yazilir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import HEADER, upsert_csv  # noqa: E402
from train import ROOT  # noqa: E402

CONF = 0.25


def hayashi(count: int) -> int:
    if count <= 5:
        return 0
    if count <= 20:
        return 1
    if count <= 50:
        return 2
    return 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="results/yolov8n_best.pt")
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--epoch", type=int, default=50)
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.is_file():
        found = sorted(
            (ROOT / "runs" / "detect").glob("*/weights/best.pt"),
            key=lambda path: path.stat().st_mtime,
        )
        if not found:
            raise SystemExit(
                "YOLO agirligi yok. runs/detect altinda best.pt bulunamadi. "
                "Sayim egitimi silmez; egitim hucresini yeniden calistir."
            )
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(found[-1].read_bytes())
        print("agirlik kopyalandi", found[-1])
    model = YOLO(str(weights))
    data = ROOT / "data" / "yolo" / "data.yaml"
    metrics = model.val(data=str(data), split="test", imgsz=640, batch=16, verbose=False)
    map50 = float(metrics.box.map50)
    upsert_csv(
        ROOT / "results" / "results.csv",
        ["yolov8n", f"{map50:.4f}", "", "", "", "", "", str(args.epoch), f"{args.minutes:.2f}"],
    )
    print(f"mAP50 {map50:.4f}")

    frame = pd.read_csv(ROOT / "data" / "splits.csv")
    test = frame[frame.kume == "test"]
    image_dir = ROOT / "data" / "yolo" / "images" / "test"
    y_true = []
    y_pred = []
    empty = 0
    for row in test.itertuples(index=False):
        prediction = model.predict(
            source=str(image_dir / row.file_name),
            conf=CONF,
            imgsz=640,
            verbose=False,
        )[0]
        count = 0 if prediction.boxes is None else len(prediction.boxes)
        if count == 0:
            empty += 1
        y_true.append(int(row.level))
        y_pred.append(hayashi(count))

    from sklearn.metrics import accuracy_score, f1_score, recall_score

    labels = [0, 1, 2, 3]
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    accuracy = float(accuracy_score(y_true, y_pred))
    score = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    upsert_csv(
        ROOT / "results" / "results.csv",
        [
            "yolov8n_sayimdan_siddet",
            f"{accuracy:.4f}",
            f"{score:.4f}",
            *[f"{float(value):.4f}" for value in recalls],
            str(args.epoch),
            f"{args.minutes:.2f}",
        ],
    )
    note = ROOT / "results" / "sifir_kutu.txt"
    note.write_text(
        f"Test goruntusu {len(test)}. Hic kutu cikmayan {empty}. Guven esigi {CONF}.\n",
        encoding="utf-8",
    )
    print(f"sayim dogruluk {accuracy:.4f} macro_f1 {score:.4f} sifir_kutu {empty}")
    print("Kisi 3 notu:", note)


if __name__ == "__main__":
    main()
