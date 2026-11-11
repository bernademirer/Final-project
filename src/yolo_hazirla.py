"""ACNE04 kutularini tek sinif YOLO etiketine cevirir. Yeni ayrim uretmez."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def yolo_line(xmin, ymin, xmax, ymax, width, height) -> str | None:
    xmin = min(max(float(xmin), 0.0), width)
    xmax = min(max(float(xmax), 0.0), width)
    ymin = min(max(float(ymin), 0.0), height)
    ymax = min(max(float(ymax), 0.0), height)
    box_w = xmax - xmin
    box_h = ymax - ymin
    if box_w <= 0 or box_h <= 0 or width <= 0 or height <= 0:
        return None
    xc = ((xmin + xmax) / 2) / width
    yc = ((ymin + ymax) / 2) / height
    return f"0 {xc:.6f} {yc:.6f} {box_w / width:.6f} {box_h / height:.6f}"


def link_image(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def main() -> None:
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    boxes = {}
    with (ROOT / "data" / "boxes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            boxes[row["file_name"]] = row

    yolo = ROOT / "data" / "yolo"
    dropped = 0
    kept = 0
    names = {split: set() for split in ("train", "val", "test")}
    for row in splits.itertuples(index=False):
        split = row.kume
        names[split].add(row.file_name)
        source = ROOT / "data" / "images" / row.file_name
        if not source.exists():
            raise SystemExit(f"goruntu yok: {source}")
        link_image(source, yolo / "images" / split / row.file_name)
        label_path = yolo / "labels" / split / f"{Path(row.file_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        record = boxes.get(row.file_name)
        lines = []
        if record:
            for xmin, ymin, xmax, ymax in zip(
                record["xmin"], record["ymin"], record["xmax"], record["ymax"]
            ):
                text = yolo_line(xmin, ymin, xmax, ymax, record["width"], record["height"])
                if text is None:
                    dropped += 1
                else:
                    lines.append(text)
                    kept += 1
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    overlap = names["train"] & names["test"]
    if overlap:
        raise SystemExit(f"train ve test ayni dosyayi paylasiyor: {len(overlap)}")
    if kept < 9000:
        raise SystemExit(f"etiket satiri cok az: {kept}. Koordinat klasoru kontrol edilmeli.")

    data = {
        "path": str((yolo).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["lezyon"],
    }
    with (yolo / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    print(f"etiket satiri {kept}")
    print(f"atilan kutu {dropped}")
    print("train", len(names["train"]), "val", len(names["val"]), "test", len(names["test"]))


if __name__ == "__main__":
    main()
