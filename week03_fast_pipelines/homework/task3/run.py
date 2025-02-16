from profiler import Profile
import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Settings, Clothes, seed_everything
from vit import ViT


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)[:100]
    val_frame = frame.drop(train_frame.index)[:100]

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", val_frame, transform=val_transforms
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=Settings.batch_size, shuffle=False)

    return train_loader, val_loader


def run_epoch(model, train_loader, val_loader, criterion, optimizer, profiler=None) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()
    for data, label in tqdm(train_loader, desc="Train"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        
        # my code
        if profiler:
            with profiler:
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
        else:
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
        # end

        optimizer.step()

        # my code
        if profiler:
            profiler.step()
        # end

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc.item() / len(train_loader)
        epoch_loss += loss.item() / len(train_loader)

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy


def main():
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)


    # my code
    profiler = Profile(model, name="ViTModel", schedule={"wait": 1, "warmup": 1, "active": 2})

    
    for epoch in range(4):
        
        print(f"Epoch {epoch}")
        epoch_loss, epoch_acc, val_loss, val_acc = run_epoch(model, train_loader, val_loader, criterion, optimizer, profiler)

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # log last epoch
    profiler.to_perfetto("custom_trace.json")

    # end


if __name__ == "__main__":
    main()
