"""ResNet50 ve EfficientNet-B0 egitimi.

    python src/train.py --model resnet50 --aug on --split data/splits.csv
    python src/train.py --model efficientnet_b0 --aug on --split data/splits.csv

Tam egitim denemesi Google Colab T4 uzerinde yapilir. Bu makinede CUDA yoktur;
CPU'da yalniz --limit-batches ile kisa duman testi calisir, agirlik yazilmaz.
aug on  -> results/{model}_best.pt
aug off -> results/{model}_aug_off.pt
--name verilirse -> results/{name}.pt, {name}_val.json, {name}_egri.png

Asama 2 (ek deney, docs/protokol_asama2.md):
    python src/train.py --model convnext_tiny --aug guclu --image-size 384 --balanced --amp --name convnext_tiny_asama2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
LABELS = [0, 1, 2, 3]
MODEL_NAMES = ["resnet50", "efficientnet_b0", "convnext_tiny", "densenet121"]


def load_cfg() -> dict:
    with (ROOT / "configs" / "ortak.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_transforms(is_train, aug_on, cfg=None, image_size=None):
    """aug_on: True/False ya da "on" / "off" / "guclu"."""
    cfg = cfg or load_cfg()
    size = int(image_size or cfg["image_size"])
    mode = aug_on if isinstance(aug_on, str) else ("on" if aug_on else "off")
    normalize = transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ops = [transforms.Resize((size, size))]
    if is_train and mode == "on":
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    if is_train and mode == "guclu":
        # Siddet etiketi lezyon sayisindan gelir: kirpma goruntunun en az %85'ini korur,
        # renk tonu (hue) kizariklik ipucunu bozmamak icin degismez.
        ops.extend(
            [
                transforms.RandomResizedCrop((size, size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            ]
        )
    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def get_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(2048, 4)
        return model
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, 4)
        return model
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(768, 4)
        return model
    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(1024, 4)
        return model
    raise ValueError("Desteklenmeyen model!")


def balanced_sampler(frame: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    counts = frame.level.value_counts()
    weights = torch.tensor([1.0 / counts[level] for level in frame.level], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(frame), replacement=True, generator=generator)


class AcneSplit(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(self.image_dir / row.file_name).convert("RGB")
        return self.transform(image), int(row.level)


def read_splits(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"file_name", "level", "kume"} - set(frame.columns)
    if missing:
        raise SystemExit(f"splits.csv eksik sutun: {sorted(missing)}")
    return frame


def check_official_split(frame: pd.DataFrame, image_dir: Path) -> None:
    if len(frame) != 1457:
        return
    counts = frame.kume.value_counts().to_dict()
    if counts.get("test") != 292 or frame.file_name.nunique() != 1457:
        raise SystemExit(
            f"splits.csv kilitli sayilarla uyusmuyor: {counts}, tekil dosya {frame.file_name.nunique()}"
        )
    missing = [name for name in frame.file_name if not (image_dir / name).exists()]
    if missing:
        raise SystemExit(f"data/images icinde {len(missing)} dosya yok. Ilk ornek: {missing[0]}")


def macro_f1(y_true, y_pred) -> float:
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))


def run_epoch(model, loader, criterion, optimizer, device, limit_batches, scaler=None):
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None
    losses = []
    y_true = []
    y_pred = []
    for step, (images, labels) in enumerate(loader):
        if limit_batches and step >= limit_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    mean_loss = sum(losses) / max(len(losses), 1)
    return mean_loss, macro_f1(y_true, y_pred)


def save_curve(history, path: Path) -> None:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(list(epochs), history["loss"], label="egitim kaybi")
    axis.plot(list(epochs), history["f1"], label="dogrulama macro-F1")
    axis.set_xlabel("epoch")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--aug", required=True, choices=["on", "off", "guclu"])
    parser.add_argument("--split", default="data/splits.csv")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--balanced", action="store_true", help="sinif dengeli ornekleme")
    parser.add_argument("--amp", action="store_true", help="karma hassasiyet (yalniz CUDA)")
    parser.add_argument("--name", default=None, help="cikti dosyalarinin adi")
    args = parser.parse_args()

    cfg = load_cfg()
    seed = int(cfg["seed"])
    set_seed(seed)
    aug = args.aug != "off"
    image_size = int(args.image_size or cfg["image_size"])
    smoke = args.limit_batches is not None
    max_epochs = args.max_epochs or int(cfg["max_epochs"])
    if smoke and args.max_epochs is None:
        max_epochs = 1

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = ROOT / image_dir

    frame = read_splits(split_path)
    check_official_split(frame, image_dir)
    train_frame = frame[frame.kume == "train"]
    train_ds = AcneSplit(train_frame, image_dir, get_transforms(True, args.aug, cfg, image_size))
    val_ds = AcneSplit(frame[frame.kume == "val"], image_dir, get_transforms(False, False, cfg, image_size))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("train veya val kumesi bos.")

    loader_kwargs = {"batch_size": int(cfg["batch_size"]), "num_workers": int(cfg["num_workers"])}
    if args.balanced:
        train_loader = DataLoader(train_ds, sampler=balanced_sampler(train_frame, seed), **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    if not torch.cuda.is_available() and not smoke:
        raise SystemExit(
            "Tam egitim denemesi bu bilgisayarda baslamaz. CUDA yok. "
            "Google Colab'da T4 GPU acip ayni komutu orada calistir."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Egitim basliyor: Model={args.model}, Augmentation={args.aug}, boyut={image_size}, "
        f"dengeli={args.balanced}, amp={args.amp}, CSV={split_path}, cihaz={device}"
    )
    model = get_model(args.model).to(device)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    if args.name:
        stem = args.name
        weight_name = f"{args.name}.pt"
    else:
        stem = args.model if aug else f"{args.model}_aug_off"
        weight_name = f"{args.model}_best.pt" if aug else f"{args.model}_aug_off.pt"
    weight_path = results / weight_name

    best_f1 = -1.0
    best_epoch = 0
    wait = 0
    patience = int(cfg["patience"])
    history = {"loss": [], "f1": []}
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, args.limit_batches, scaler)
        _, val_f1 = run_epoch(model, val_loader, criterion, None, device, args.limit_batches, scaler)
        history["loss"].append(loss)
        history["f1"].append(val_f1)
        print(f"epoch {epoch} kayip {loss:.4f} val_macro_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            wait = 0
            if not smoke:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "minutes": (time.perf_counter() - started) / 60,
                        "val_macro_f1": val_f1,
                        "model": args.model,
                        "aug": args.aug,
                        "image_size": image_size,
                        "balanced": args.balanced,
                    },
                    weight_path,
                )
        else:
            wait += 1
            if wait >= patience:
                print("erken durdurma")
                break

    if smoke:
        print("duman testi bitti. agirlik ve sonuc tablosu yazilmadi.")
        return

    save_curve(history, results / f"{stem}_egri.png")
    payload = {"val_macro_f1": best_f1, "epoch": best_epoch}
    (results / f"{stem}_val.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"en iyi val macro-F1 {best_f1:.4f} epoch {best_epoch}")
    print("agirlik", weight_path)


if __name__ == "__main__":
    main()
