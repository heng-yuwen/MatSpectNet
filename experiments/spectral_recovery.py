"""
This experiment implements the method to recover hyper-spectral images from rgb images.
Models are taken from  MST-plus-plus
https://github.com/caiyuanhao1998/MST-plus-plus
"""

import pytorch_lightning as pl
from models import spectral_recovery_models
from customise_pl.losses import Loss_MRAE, Loss_RMSE, Loss_PSNR
from torchmetrics import MeanMetric
from customise_pl.schedulers import build_scheduler


class SpectralRecovery(pl.LightningModule):
    def __init__(self, method: str, optimizer_dict: dict, scheduler_dict: dict, pretrained_model_path=None, ):
        super().__init__()
        self.automatic_optimization = False
        self.criterion_mrae = Loss_MRAE()
        self.criterion_rmse = Loss_RMSE()
        self.criterion_psnr = Loss_PSNR()
        self.model = spectral_recovery_models.get_models(method, pretrained_model_path=pretrained_model_path)
        self.mean_mrae_loss = MeanMetric()
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict

    def forward(self, data):
        # in lightning,
        # forward defines the prediction/inference actions
        return self.model(data)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # It is independent of forward
        data, target = batch
        preds = self.model(data)
        loss = self.criterion_mrae(preds, target)

        # train
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        # scheduler
        scheduler = self.lr_schedulers()
        self.lr_scheduler_step(scheduler)

        return {'loss': loss, 'preds': preds, 'target': target}

    def lr_scheduler_step(self, scheduler, optimizer_idx=None, metric=None):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def training_step_end(self, outputs):
        self.log("train_loss", outputs["loss"], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        data, target = batch
        preds = self.model(data)
        loss_mrae = self.criterion_mrae(preds, target)
        loss_rmse = self.criterion_rmse(preds, target)
        loss_psnr = self.criterion_psnr(preds, target)
        return {'loss_mrae': loss_mrae, 'loss_rmse': loss_rmse, 'loss_psnr': loss_psnr}

    def validation_step_end(self, outputs):
        self.log("loss_mrae", outputs["loss_mrae"])
        self.log("loss_rmse", outputs["loss_rmse"])
        self.log("loss_psnr", outputs["loss_psnr"])
        self.mean_mrae_loss.update(outputs["loss_mrae"])

    def on_validation_epoch_end(self):
        avg_mrae_loss = self.mean_mrae_loss.compute()
        self.mean_mrae_loss.reset()
        self.log("mean_mrae_loss", avg_mrae_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        data = batch
        preds = self.model(data)
        return {'preds': preds}

    def test_step_end(self, outputs):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        from mmcv.runner import build_optimizer
        optimizer = build_optimizer(model=self.model, cfg=self.optimizer_dict)
        scheduler = build_scheduler(optimizer=optimizer, cfg=self.scheduler_dict, num_epochs=self.trainer.max_epochs,
                                    num_training_steps=224)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
