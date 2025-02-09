import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel
import wandb
from hparams import config


def main(device: str, config=config):
    wandb.init(
        project="effdl",
        config=config,
    )


    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=config['hidden_size']),
        betas=config['betas'],
        num_timesteps=config['num_timesteps'],
    )
    ddpm.to(device)


    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=4, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=config['learning_rate'])

    for epoch in range(config["num_epochs"]):
        
        sample_path = f"samples/{epoch:02d}.png"
        epoch_loss = train_epoch(ddpm, dataloader, optim, device)

        generate_samples(ddpm, device, sample_path)


        wandb.log({
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "learning_rate": optim.param_groups[0]['lr'],
            "Generated Samples": wandb.Image(sample_path)
        })


if __name__ == "__main__":
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    main(device=device)
