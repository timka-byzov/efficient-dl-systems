import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


import torch
from enum import Enum


class LossScaler:

    def __init__(self, scale=1):
        self.cur_scale = scale

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        return False

    # `overflow` is boolean indicating whether we overflowed in gradient
    def update_scale(self, overflow):
        pass

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()

class DynamicLossScaler:

    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
#        return False
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        try:
            # Stopgap until upstream fixes sum() on HalfTensors
            cpu_sum = float(x.float().sum())
            # cpu_sum = float(x.sum())
            # print(cpu_sum)
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether we overflowed in gradient
    def update_scale(self, overflow):
        if overflow:
            #self.cur_scale /= self.scale_factor
            self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                self.cur_scale *= self.scale_factor
#        self.cur_scale = 1
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()



def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    loss_scaler: DynamicLossScaler | LossScaler | None,
    device: torch.device,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        # TODO: your code for loss scaling here
        scaled_loss = loss * loss_scaler.loss_scale

        optimizer.zero_grad()
        scaled_loss.backward()

        params = model.parameters()
        # Check for overflow
        has_overflow = loss_scaler.has_overflow(params)
        
        # If no overflow, unscale grad and update as usual
        if not has_overflow:
            for param in  params:
                param.grad.data.mul_(1. / loss_scaler.loss_scale)
            optimizer.step()

        
        loss_scaler.update_scale(has_overflow)

        # end of my code

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(use_dynamic_scaling: bool = False):
    device = torch.device("cuda:4")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()


    if use_dynamic_scaling:
        loss_scaler = LossScaler()
    else:
        loss_scaler = DynamicLossScaler()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, loss_scaler, device=device)



if __name__ == "__main__":
    train()
