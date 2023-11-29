import torch
from torchmetrics.functional.classification import dice
from torchmetrics.functional.detection import intersection_over_union
from torchmetrics.image import StructuralSimilarityIndexMeasure


class MetricsUnet:
    def __init__(self) -> None:
        self.train_correct_num = 0
        self.train_total = 0

        self.intersection = 0
        self.union = 0

    def accuracy(self, preds, target):
        predictions = torch.argmax(preds, dim=1)
        self.train_correct_num += (predictions == target).sum()
        self.train_total += torch.numel(predictions)
        acc = float(self.train_correct_num) / self.train_total

        return acc

    @staticmethod
    def ssim(preds, target):
        ssim = StructuralSimilarityIndexMeasure()
        return ssim(preds, target).item()

    @staticmethod
    def dice(preds, target):
        return dice(preds, target).item()

    def iou(self, preds, target):
        predictions = torch.argmax(preds, dim=1)
        self.intersection += (predictions * target).sum()
        self.union += (predictions + target - predictions * target).sum()
        iou = (self.intersection) / (self.union + 1e-8)
        return iou


# class Accuracy:
#     @staticmethod
#     def accuracy(model, data_loader, device):
#         dice = 0.0
#         iou = 0.0
#         val_loss = 0.0
#         val_acc = 0.0

#         with torch.no_grad():
#             correct = 0
#             intersection = 0
#             denom = 0
#             union = 0
#             total = 0
#             cost = 0.0

#             for index, (features, labels) in enumerate(data_loader, start=1):
#                 # Data
#                 x = features.to(device=device, dtype=torch.float32)
#                 # y = labels.to(device=device, dtype=torch.long).squeeze(1)

#                 # predictions
#                 bottleneck, output = model(x)

#                 # Compute the loss and its gradients
#                 cost = F.mse_loss(input=output, target=x)

#                 # standard accuracy and loss
#                 # preds = torch.argmax(output, dim=0)
#                 correct += (output == x).sum()
#                 total += torch.numel(output)

#                 val_loss = cost / len(data_loader)
#                 val_acc = float(correct) / total

#                 # dice coefficient
#                 intersection += (output * x).sum()
#                 denom += (output + x).sum()
#                 dice = 2 * intersection / (denom + 1e-8)

#                 # intersection over union
#                 union += (output + x - output * x).sum()
#                 iou = (intersection) / (union + 1e-8)

#             del correct, intersection, denom, union, total, cost

#         return val_loss, val_acc, dice, iou
