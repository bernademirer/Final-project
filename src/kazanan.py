"""Kazanan modeli yalniz dogrulama macro-F1 ile secer.

ResNet50 val json yoksa dosya yazilmaz. Esitlikte resnet50 secilir.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results = ROOT / "results"
    resnet_path = results / "resnet50_val.json"
    eff_path = results / "efficientnet_b0_val.json"
    if not resnet_path.exists() or not eff_path.exists():
        missing = [path.name for path in (resnet_path, eff_path) if not path.exists()]
        print("kazanan.txt yazilmadi. Eksik:", ", ".join(missing))
        return
    resnet = json.loads(resnet_path.read_text(encoding="utf-8"))["val_macro_f1"]
    eff = json.loads(eff_path.read_text(encoding="utf-8"))["val_macro_f1"]
    if eff > resnet:
        winner = "efficientnet_b0"
    else:
        winner = "resnet50"
    (results / "kazanan.txt").write_text(winner + "\n", encoding="utf-8")
    print(f"resnet50 {resnet:.4f} efficientnet_b0 {eff:.4f} kazanan {winner}")
    if resnet == eff:
        print("skorlar esit, ResNet50 secildi")


if __name__ == "__main__":
    main()
