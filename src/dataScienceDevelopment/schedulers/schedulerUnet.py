from torch.optim import lr_scheduler


class SchedulerUnet:
    def __init__(self, optimizer, epochs) -> None:
        self.optimizer = optimizer
        self.epochs = epochs

    def scheduler_OneCycleLR(self, steps_per_epoch, max_lr: float = 1e-1):
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=max_lr,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.43,
            div_factor=10,
            final_div_factor=1e3,
            three_phase=True,
        )

        return scheduler
