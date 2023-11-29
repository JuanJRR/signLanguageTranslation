import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dvclive import Live

from ..metrics.metricsUnet import MetricsUnet
from ..utils.trainingReportUnet import TrainingReportUnet


class UnetTraining:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_function,
        data_training,
        model_dimensions: dict = {},
        data_validation=None,
        epochs: int = 4,
        scheduler=None,
        path_save_experiment: str = "",
        early_stop: dict = {"stop": True, "patience": 5},
        details: dict = {"report": None, "iterations": None},
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # Settings
        self.model_dimensions = model_dimensions
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure()

        # Model
        self.model = model

        # Optimizer
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training
        self.epochs = epochs
        self.early_stop = early_stop
        self.best_acc = 0.0
        self.patience = 0

        # Data
        self.data_training = data_training
        self.data_validation = data_validation

        # Reports
        self.details = details
        self.path_save_experiment = path_save_experiment
        self.training_report_unet = TrainingReportUnet(
            report=details["report"], iterations=details["iterations"]
        )

        # Metrics
        self.metrics = MetricsUnet()

    def __training_one_epoch(self, model: nn.Module, data_training, epoch):
        # Settings
        running_loss = 0.0
        last_loss = 0.0

        running_acc = 0.0
        last_acc = 0.0

        model.train(True)
        model.to(device=self.device)

        # Loop
        for index, (features, labels) in enumerate(data_training, start=1):
            last_acc = 0.0
            last_loss = 0.0

            # Data
            x = features.to(device=self.device, dtype=torch.float32)
            y = labels.to(device=self.device, dtype=torch.long).squeeze(1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Predictions
            bottleneck, output = model(x)

            # Compute the loss and its gradients
            loss_bottleneck = torch.tensor(0.0)
            loss_output = self.loss_function(input=output, target=y)

            loss = loss_bottleneck + loss_output
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Learning rate adjustment
            if self.scheduler:
                self.scheduler.step()

            # Metrics
            running_loss += loss.item()
            last_loss = running_loss / index

            running_acc += self.metrics.accuracy(output, y)
            last_acc = running_acc / index

            # Report
            self.training_report_unet.report_one_epoch(
                epoch=epoch, index=index, last_loss=last_loss, last_acc=last_acc
            )

            del x, y, output, bottleneck
        del running_loss, running_acc

        return last_loss, last_acc

    def __validation_one_epoch(self, model: nn.Module, data_validation):
        # Settings
        running_loss = 0.0
        last_loss = 0.0

        running_acc = 0.0
        last_acc = 0.0

        dice = 0.0
        iou = 0.0

        model.eval()
        model.to(device=self.device)

        # Loop
        with torch.no_grad():
            for index, (features, labels) in enumerate(data_validation, start=1):
                last_acc = 0.0
                last_loss = 0.0

                # Data
                x = features.to(device=self.device, dtype=torch.float32)
                y = labels.to(device=self.device, dtype=torch.long).squeeze(1)

                # Predictions
                bottleneck, output = model(x)

                # Compute the loss and its gradients
                loss_bottleneck = torch.tensor(0.0)
                loss_output = self.loss_function(input=output, target=y)

                loss = loss_bottleneck + loss_output

                # Metrics
                running_loss += loss.item()
                last_loss = running_loss / index

                running_acc += self.metrics.accuracy(output, y)
                last_acc = running_acc / index

                dice = self.metrics.dice(output, y)
                iou = self.metrics.iou(output, y)

                del x, y, bottleneck, output
            del running_loss, running_acc

            return last_loss, last_acc, dice, iou

    def __checkpoints(self, name, epoch, train_loss, train_acc, val_loss, val_acc, acc):
        path_checkpoints = os.path.join(
            self.path_save_experiment,
            f"checkpoints_e{epoch}_{name}_acc{int(acc*100)}.pt",
        )
        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            f=path_checkpoints,
        )

    def __early_stop(self, name, val_acc):
        path_model = os.path.join(self.path_save_experiment, f"{name}_model.pt")

        if self.early_stop["stop"] == True:
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience = 0

                torch.save(obj=self.model.state_dict(), f=path_model)
            else:
                self.patience += 1
        elif self.early_stop["stop"] == False:
            self.patience = 0
            torch.save(obj=self.model.state_dict(), f=path_model)

    def training_loop(self, name):
        val_loss = 0.0
        val_acc = 0.0
        dice = 0.0
        iou = 0.0

        with Live(dvcyaml="experiments/unet/dvc.yaml") as live:
            params = {
                "metrics": ["loss", "accuracy"],
                "training": {
                    "optimizer": self.optimizer.state_dict(),
                    "model_dimensions": self.model_dimensions,
                },
            }

            live.log_params(params)

            for epoch in range(self.epochs):
                # Training
                train_loss, train_acc = self.__training_one_epoch(
                    model=self.model, data_training=self.data_training, epoch=epoch
                )
                # Validation
                if self.data_validation:
                    val_loss, val_acc, dice, iou = self.__validation_one_epoch(
                        model=self.model, data_validation=self.data_validation
                    )
                    # Checkpoints
                    self.__checkpoints(
                        name=name,
                        epoch=epoch,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        acc=val_acc,
                    )
                else:
                    val_loss = 0.0
                    val_acc = 0.0

                    # Checkpoints
                    self.__checkpoints(
                        name=name,
                        epoch=epoch,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        acc=train_acc,
                    )

                # Save model
                if self.patience == self.early_stop["patience"]:
                    break
                else:
                    self.__early_stop(name=name, val_acc=val_acc)

                # Report
                live.log_param("epochs", epoch)

                live.log_metric("train/loss", float(train_loss))
                live.log_metric("train/accuracy", float(train_acc))

                live.log_metric("validation/loss", float(val_loss))
                live.log_metric("validation/accuracy", float(val_acc))

                live.log_metric("validation/dice", float(dice))
                live.log_metric("validation/iou", float(iou))

                live.next_step()

                self.training_report_unet.report_training_loop(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    dice=dice,
                    iou=iou,
                )
