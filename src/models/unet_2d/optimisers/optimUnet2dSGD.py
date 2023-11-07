import torch.nn as nn
from torch import optim


class OptimUnet2dSGD:
    @staticmethod
    def optim_sgd_1(
        model: nn.Module, lr: float, momentum: float = 0, weight_decay: float = 0
    ) -> optim.Optimizer:
        optim_sgd = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        return optim_sgd
