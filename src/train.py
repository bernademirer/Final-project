import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_transforms(is_train, aug_on):
    # Temel dönüşümler (Her durumda geçerli)
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Sadece eğitimde ve aug_on ise artırma ekle
    if is_train and aug_on:
        aug_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ]
        return transforms.Compose(aug_transforms + base_transforms)
    
    return transforms.Compose(base_transforms)

def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(2048, 4)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(1280, 4)
    else:
        raise ValueError("Desteklenmeyen model!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['resnet50', 'efficientnet_b0'], required=True)
    parser.add_argument('--aug', type=str, choices=['on', 'off'], required=True)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    print(f"Eğitim başlıyor: Model={args.model}, Augmentation={args.aug}, CSV={args.split}")
    
    # Model ve transformları başlat
    model = get_model(args.model)
    train_transform = get_transforms(is_train=True, aug_on=(args.aug == 'on'))
    val_transform = get_transforms(is_train=False, aug_on=False)

    # TODO: PyTorch Dataset sınıfını yaz (splits.csv okuyacak şekilde)
    # TODO: Eğitim döngüsünü (epoch, erken durdurma, macro-F1 hesaplama) ekle