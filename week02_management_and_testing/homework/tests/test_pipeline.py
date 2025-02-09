import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5




@pytest.fixture
def sample_output_file(tmp_path):
    file_path = tmp_path / "test_samples_output.png"
    yield file_path
    # Cleanup: remove the file if it exists after the test.
    if file_path.exists():
        file_path.unlink()

@pytest.mark.parametrize("learning_rate, num_epochs, expected_loss_threshold", [
    (1e-3, 1, 2.0),
    (5e-4, 2, 1.5),
    (1e-4, 3, 1.0),
])
@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(learning_rate, num_epochs, expected_loss_threshold, sample_output_file, device):

    # 1. Сгенерируем синтетический датасет
    inputs = torch.randn(32, 3, 32, 32)
    labels = torch.zeros(32)    # фикс даталоадера
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Инициализация
    unet = UnetModel(in_channels=3, out_channels=3, hidden_size=32)
    model = DiffusionModel(
        eps_model=unet,
        betas=(1e-4, 2e-2),
        num_timesteps=10
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 3. Обучим модель
    for _ in range(num_epochs):
        train_epoch(model, dataloader, optimizer, device)

    # 4. Проверим, что работает генерация
    generate_samples(model, device, str(sample_output_file))

    # 5. Чекнем лосс на одном батче
    test_batch = next(iter(dataloader))[0].to(device)

    with torch.no_grad():
        final_loss = model(test_batch).item()

    assert final_loss < expected_loss_threshold, (
        f"Final loss {final_loss:.4f} did not meet the expected threshold {expected_loss_threshold:.4f}"
    )