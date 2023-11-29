import torch.nn as nn
from torch import optim


class OptimUnet:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def sgd(
        self, lr: float = 0.001, momentum: float = 0.95, weight_decay: float = 1e-4
    ):
        optim_sgd = optim.SGD(
            params=self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        return optim_sgd

    def adam(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        optim_adam = optim.Adam(
            params=self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

        return optim_adam
