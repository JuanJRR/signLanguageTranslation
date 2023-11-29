class TrainingReportUnet:
    def __init__(self, report=None, iterations=None) -> None:
        self.report = report
        self.iterations = iterations
        self.last_epoch = 0
        self.flag_epoch = True

    def __epoch(self, epoch):
        if self.last_epoch != epoch:
            self.last_epoch = epoch
            print(f" ----- EPOCH: {epoch+1} ----- ")
        elif self.flag_epoch:
            self.flag_epoch = False
            print(f" ----- EPOCH: {epoch+1} ----- ")

    def report_one_epoch(self, epoch, index, last_loss, last_acc):
        if self.report == 1:
            self.__epoch(epoch=epoch)
            if index % self.iterations == 0:
                print(
                    f"    batch: {index} -> loss: {last_loss:.8f} -- acc (ssim): {last_acc:.8f}"
                )

    def report_training_loop(
        self, epoch, train_loss, val_loss, train_acc, val_acc, dice, iou
    ):
        if self.report == 1:
            self.__epoch(epoch=epoch)
            print(f"train_loss: {train_loss:.8f} -- val_loss: {val_loss:.8f}")
            print(f"train_acc: {train_acc:.8f} -- val_acc: {val_acc:.8f}")
            print(f"dice: {dice:.8f} -- iou: {iou:.8f}")

        elif self.report == 2:
            self.__epoch(epoch=epoch)
            print(f"train_loss: {train_loss:.8f} -- val_loss: {val_loss:.8f}")
            print(f"train_acc: {train_acc:.8f} -- val_acc: {val_acc:.8f}")
            print(f"dice: {dice:.8f} -- iou: {iou:.8f}")
