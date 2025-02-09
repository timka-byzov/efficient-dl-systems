from omegaconf import DictConfig, OmegaConf
import torch
import hydra
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, generate_samples_from_batch, train_epoch
from modeling.unet import UnetModel
import wandb


import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from hydra import initialize, compose
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

def save_config_artifact(wandb_run, cfg):

    artifact = wandb.Artifact(
        name="hydra_config", 
        type="config",
        description="Hydra configuration for this run",
        metadata={
            "optimizer": cfg.optimizer,
            "learning_rate": cfg.trainer.peak_lr,
            "momentum": cfg.trainer.momentum if cfg.optimizer == "sgd" else "N/A",
            "batch_size": cfg.trainer.batch_size,
            "epochs": cfg.trainer.num_epochs,
            "flip_augment": cfg.trainer.flip_augment,
            "num_workers": cfg.trainer.num_workers
        }
    )

    config_path = "hydra_config.yaml"
    OmegaConf.save(config=cfg, f=config_path)

    artifact.add_file(config_path)

    wandb_run.log_artifact(artifact)


def get_optimizer(model, cfg):
    optimiser_config = cfg.optimizer_configs[cfg.optimizer]
    if optimiser_config.name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.trainer.peak_lr,
            momentum=optimiser_config.momentum,
        )
    else:
        # Default: Adam
        return torch.optim.Adam(model.parameters(), lr=cfg.trainer.peak_lr, weight_decay=optimiser_config.weight_decay)


def get_data(cfg):
    transform_list = []
    if cfg.trainer.flip_augment:
        transform_list.append(transforms.RandomHorizontalFlip())

    # Always normalize for CIFAR (example)
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = transforms.Compose(transform_list)

    train_dataset = CIFAR10(
        root=to_absolute_path("./data"), train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
    )
    return train_dataset, train_loader


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    device = cfg.trainer.device

    run = wandb.init(project="effdl", config=OmegaConf.to_container(cfg, resolve=True))

    save_config_artifact(run, cfg)

    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=cfg.trainer.hid_size),
        betas=(cfg.trainer.beta1, cfg.trainer.beta2),
        num_timesteps=cfg.trainer.num_timesteps,
    )
    ddpm.to(device)

    dataset, dataloader = get_data(cfg)
    sample_batch, _ = next(iter(DataLoader(dataset, batch_size=cfg.trainer.batch_size)))
    wandb.log({f"sample batch": [wandb.Image(sample_batch)]})

    optim = get_optimizer(ddpm, cfg)

    for epoch in range(cfg.trainer.num_epochs):

        sample_path = f"samples/{epoch:02d}.png"
        epoch_loss = train_epoch(ddpm, dataloader, optim, device)

        generate_samples(ddpm, device, sample_path)

        wandb.log(
            {
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "learning_rate": optim.param_groups[0]["lr"],
                "generated samples from batch": wandb.Image(
                    generate_samples_from_batch(ddpm, sample_batch.to(device))
                ),
                "generated samples": wandb.Image(sample_path),
            }
        )


if __name__ == "__main__":
    main()
