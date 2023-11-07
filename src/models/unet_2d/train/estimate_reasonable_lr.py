import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from utils.logger import Logger


class EstimateReasonableLr:
    @staticmethod
    def estimate_lr(
        model: nn.Module,
        optim: optim.Optimizer,
        data_loader: DataLoader,
        max_lr: float = 1,
        min_lr: float = 1e-6,
        beta: float = 0.99,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        # seingen
        logger = Logger()
        log = logger.config_logging()

        iterations = len(data_loader)
        update_factor = (max_lr / min_lr) ** (1 / iterations)

        lr = min_lr
        lrs = []

        loss = 0.0
        lowest_loss = 0.0
        average_loss = 0.0
        losses = []

        acc = 0.0
        # lowest_acc = 0.0
        average_acc = 0.0
        accuracies = []

        optim.param_groups[0]["lr"] = lr
        model = model.to(device=device)

        for i, (features, labels) in enumerate(data_loader, start=1):
            x = features.to(device=device, dtype=torch.float32)
            # y = labels.to(device=device, dtype=torch.long).squeeze(1)

            optim.zero_grad()

            bottleneck, output = model(x)

            cost = F.cross_entropy(input=output, target=x)

            # weighted exponential average loss
            loss = beta * loss + (1 - beta) * cost.item()
            average_loss = loss / (1 - beta**i)

            # weighted exponential average accuracy
            preds = torch.argmax(output, dim=0)
            acc_ = (preds == x).sum() / torch.numel(output)
            acc = beta * acc + (1 - beta) * acc_.item()
            average_acc = acc / (1 - beta**i)

            if i > 1 and average_loss > 4 * lowest_loss:
                print(f"from here{i, cost.item()}")
                return lrs, losses, accuracies
            elif average_loss < lowest_loss or i == 1:
                lowest_loss = average_loss

            # logs
            accuracies.append(average_acc)
            losses.append(average_loss)
            lrs.append(lr)

            # step
            cost.backward()
            optim.step()

            # update lr
            print(f"cost:{cost.item():.4f}, lr: {lr:.6f}, acc: {acc_.item():.6f}")
            log.debug(f"cost:{cost.item():.4f}, lr: {lr:.6f}, acc: {acc_.item():.6f}")
            lr = lr * update_factor
            optim.param_groups[0]["lr"] = lr

        return lrs, losses, accuracies
