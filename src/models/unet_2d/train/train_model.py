import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dvclive import Live

from ..metrics.accuracy import Accuracy


class TrainModel:
    def __init__(
        self,
        model,
        optimizer,
        data_train,
        data_validation,
        path_save,
        batch_size=32,
        scheduler=None,
        epochs=100,
        early_stop_thresh=5,
        iterations_report=25,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        report=True,
    ) -> None:
        # settings
        self.device = device
        self.report = report
        self.path_save = path_save
        self.early_stop_thresh = early_stop_thresh

        self.epochs = epochs
        self.iterations_report = iterations_report
        self.batch_size = batch_size

        # Model
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Data
        self.dl_train = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        self.dl_val = DataLoader(
            data_validation, batch_size=self.batch_size, shuffle=True
        )

    def __train_one_epoch(self, model, data_loader):
        # settings
        running_loss = 0.0
        last_loss = 0.0

        running_acc = 0.0
        last_acc = 0.0
        preds_total = 0.0

        loss = []
        acc = []

        for index, (features, labels) in enumerate(data_loader, start=1):
            last_acc = 0.0
            last_loss = 0.0

            # Data
            x = features.to(device=self.device, dtype=torch.float32)
            # y = labels.to(device=self.device, dtype=torch.long).squeeze(1)

            # Zero gradients
            self.optimizer.zero_grad()

            # predictions
            bottleneck, output = model(x)

            # Compute the loss and its gradients
            cost = F.mse_loss(input=output, target=x)
            cost.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Learning rate adjustment
            if self.scheduler:
                self.scheduler.step()

            # Gather data and report
            running_loss += cost.item()

            # preds = torch.argmax(output, dim=0)
            preds_total += torch.numel(output)
            running_acc += (output == x).sum()

            last_loss = running_loss / index
            last_acc = running_acc / preds_total

            loss.append(last_loss)
            acc.append(last_acc)

            if index % self.iterations_report == 0 and self.report:
                print(
                    f"    batch: {index+1} -> loss: {last_loss:.8f} -- acc: {last_acc:.8f}"
                )

        train_loss = np.mean(loss)
        train_acc = np.mean(acc)

        del running_loss, last_loss, running_acc, last_acc, preds_total, loss, acc

        return train_loss, train_acc

    def train(self):
        count_early_stop_thresh = 0
        model = self.model

        path_model = os.path.join(self.path_save, "model.pt")

        best_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        dice = 0.0
        iou = 0.0

        with Live(dvcyaml="experiments/unet/dvc.yaml") as live:
            params = {
                "metrics": ["loss", "accuracy", "dice", "iou"],
                "training": {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "optimizer": self.optimizer.state_dict(),
                },
            }
            live.log_params(params)

            for epoch in range(self.epochs):
                path_checkpoints = os.path.join(
                    self.path_save, "checkpoints_" + str(epoch) + ".pt"
                )

                if self.report:
                    print(f" ----- EPOCH: {epoch +1} ----- ")

                # model training
                model = model.train(True)
                train_loss, train_acc = self.__train_one_epoch(
                    model=model, data_loader=self.dl_train
                )

                live.log_metric("train/loss", float(train_loss))
                live.log_metric("train/accuracy", float(train_acc))

                # statistics
                model = model.eval()
                val_loss, val_acc, dice, iou = Accuracy.accuracy(
                    model=model, data_loader=self.dl_val, device=self.device
                )

                live.log_metric("validation/loss", float(val_loss))
                live.log_metric("validation/accuracy", float(val_acc))
                live.log_metric("validation/dice", float(dice))
                live.log_metric("validation/iou", float(iou))

                if self.report:
                    print(f"train_loss: {train_loss:.8f} -- val_loss: {val_loss:.8f}")
                    print(f"train_acc: {train_acc:.8f} -- val acc: {val_acc:.8f}")
                    print(f"dice: {dice}, iou: {iou}")

                # checkpoints
                torch.save(
                    obj={
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                    },
                    f=path_checkpoints,
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    count_early_stop_thresh = 0

                    torch.save(
                        obj=model.state_dict(),
                        f=path_model,
                    )
                    print("save model")
                else:
                    count_early_stop_thresh += 1

                if count_early_stop_thresh > self.early_stop_thresh:
                    break

                # tracking
                # live.log_artifact(path_checkpoints)
                live.next_step()
            # live.log_artifact(path_model)

            return model
