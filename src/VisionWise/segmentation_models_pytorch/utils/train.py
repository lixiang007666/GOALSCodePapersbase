import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
import segmentation_models_pytorch as smp

from torch.cuda.amp import autocast as autocast, GradScaler
from torch.nn import CrossEntropyLoss
import apex


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        for ls in self.loss:
            ls.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'Dice': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, scheduler,device="cpu", verbose=True, amp=False, scaler=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp
        self.scaler = scaler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        if self.amp == True:
            with autocast(enabled=True):
                prediction = self.model.forward(x)
                loss_total = 0
                for ls in self.loss:
                    if isinstance(ls,smp.losses.SoftCrossEntropyLoss) or isinstance(ls,CrossEntropyLoss):
                        loss_total += ls(prediction,torch.argmax(y,dim=1))
                    else:
                        loss_total += ls(prediction, y)
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            prediction = self.model.forward(x)
            loss_total = 0
            for ls in self.loss:
                if isinstance(ls,smp.losses.SoftCrossEntropyLoss) or isinstance(ls,CrossEntropyLoss):
                    loss_total += ls(prediction,torch.argmax(y,dim=1))
                else:
                    loss_total += ls(prediction, y)
            self.optimizer.step()
            
        return loss_total, prediction

class PseudoEpoch(Epoch):
    def __init__(self, model, loss, weight,metrics, optimizer, scheduler,device="cpu", verbose=True, amp=False,scaler=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight = weight
        self.scaler = scaler
        self.amp = amp

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        if self.amp == True:
            with autocast(enabled=True):
                prediction = self.model.forward(x)
                loss_total = 0
                for ls in self.loss:
                    if isinstance(ls,smp.losses.SoftCrossEntropyLoss) or isinstance(ls,CrossEntropyLoss):
                        loss_total += ls(prediction,torch.argmax(y,dim=1))
                    else:
                        loss_total += ls(prediction, y)
                loss_total = loss_total * self.weight
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            prediction = self.model.forward(x)
            loss_total = 0
            for ls in self.loss:
                if isinstance(ls,smp.losses.SoftCrossEntropyLoss) or isinstance(ls,CrossEntropyLoss):
                    loss_total += ls(prediction,torch.argmax(y,dim=1))
                else:
                    loss_total += ls(prediction, y)
            loss_total = loss_total * self.weight
            self.optimizer.step()
        return loss_total, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss_total = 0
            for ls in self.loss:
                if isinstance(ls,smp.losses.SoftCrossEntropyLoss) or isinstance(ls,CrossEntropyLoss):
                    loss_total += ls(prediction,torch.argmax(y,dim=1))
                else:
                    loss_total += ls(prediction, y)
        return loss_total, prediction

class InferEpoch(Epoch):
    def __init__(self, model, device="cpu", verbose=True):
        super().__init__(
            model=model,
            stage_name="infer",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
        return prediction
