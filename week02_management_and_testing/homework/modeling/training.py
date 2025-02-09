import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from modeling.diffusion import DiffusionModel


def train_step(model: DiffusionModel, inputs: torch.Tensor, optimizer: Optimizer, device: str):
    optimizer.zero_grad()
    inputs = inputs.to(device)
    loss, _ = model(inputs)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: DiffusionModel, dataloader: DataLoader, optimizer: Optimizer, device: str):
    model.train()
    pbar = tqdm(dataloader)
    loss_ema = None
    for x, _ in pbar:
        train_loss = train_step(model, x, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
    return loss_ema


def generate_samples(model: DiffusionModel, device: str, path: str):
    model.eval()
    with torch.no_grad():
        samples = model.sample(8, (3, 32, 32), device=device)
        grid = make_grid(samples, nrow=4)
        save_image(grid, path)



# def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
#     """
#     Reverse the normalization applied during preprocessing.
#     """
#     mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
#     std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
#     return tensor * std + mean  # Reverse normalization

def generate_samples_from_batch(model: DiffusionModel, batch: torch.tensor):
    """
    Generate samples from a given batch using the diffusion model.
    """
    model.eval()
    with torch.no_grad():
        _, out = model(batch)  # Assuming model outputs image-like tensors
    return out
