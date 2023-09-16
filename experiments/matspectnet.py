"""
This experiment implements the method to adapt hsi recovery in material segmentation.
Instead of using synthetic data, we use domain adaptation.
We also consider the effect of transfer learning.
"""
from math import ceil
import pytorch_lightning as pl
import segmentation_models_pytorch.losses as losses
import matplotlib.pyplot as plt
from models import spectral_recovery_models
from customise_pl.losses import Loss_MRAE
from customise_pl.metrics import SegmentEvaluator
from torchmetrics.classification import MulticlassConfusionMatrix
from models.spectral_recovery_models.gan_networks import GANLoss
from models import segmentation_models

from torchvision.transforms.functional import crop
from datasets.localmatdb import lmd_labels_to_color
import torch.nn.functional as F
import torch
from datasets.localmatdb import LocalMatDataset


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_patch_info(patch_size, H, W):
    num_H = H // patch_size if H % patch_size == 0 else H // patch_size + 1
    num_W = W // patch_size if W % patch_size == 0 else W // patch_size + 1

    stride_H = patch_size if H % patch_size == 0 else ceil((H - patch_size) / (num_H - 1))
    stride_W = patch_size if W % patch_size == 0 else ceil((W - patch_size) / (num_W - 1))

    H_padded = stride_H * (num_H - 1) + patch_size
    W_padded = stride_W * (num_W - 1) + patch_size

    pad_H = H_padded - H
    pad_W = W_padded - W

    return pad_H, pad_W, stride_H, stride_W, H_padded, W_padded


def patch2global(tensor_patches, count_masks, patch_count, resized_size, patch_size=512):
    # restore global to original size
    merged_tensors = []
    sum = 0
    for idx, count in enumerate(patch_count):
        tensors = tensor_patches[sum: sum + count]
        sum += count
        H, W = resized_size[idx]
        pad_H, pad_W, stride_H, stride_W, H_padded, W_padded = get_patch_info(patch_size, H, W)
        C = tensors.size()[1]
        count_mask = count_masks[idx]

        tensors = tensors.permute(1, 2, 3, 0)
        count_mask = count_mask.permute(1, 2, 0)

        assert count_mask.size() == tensors.size()[1:], "the number of patches do not match"

        tensors = tensors.contiguous().view(C * patch_size ** 2, -1)
        count_mask = count_mask.contiguous().view(patch_size ** 2, -1)

        tensors = F.fold(tensors.unsqueeze(dim=0), output_size=(H_padded, W_padded),
                         kernel_size=patch_size, stride=(stride_H, stride_W))
        count_mask = F.fold(count_mask.unsqueeze(dim=0), output_size=(H_padded, W_padded), kernel_size=patch_size,
                            stride=(stride_H, stride_W))

        tensors = tensors / count_mask
        tensors = crop(tensors, pad_H // 2 + pad_H % 2, pad_W // 2 + pad_W % 2, H, W)
        assert tensors.size(-1) == W and tensors.size(-2) == H, "Wrong cropped region. {} does not match {}".format(
            tensors.size(), (H, W))
        merged_tensors.append(tensors)

    return merged_tensors


def split_overlap_img_tensor(images_tensor, patch_size=512):
    splitted_tensors = []
    count_mask_tensors = []
    patch_count = []
    resized_size = []
    for img_tensor in images_tensor:
        img_tensor = img_tensor.unsqueeze(0)
        _, C, H, W = img_tensor.size()
        pad_H, pad_W, stride_H, stride_W, H_padded, W_padded = get_patch_info(patch_size, H, W)
        resized_size.append((H, W))
        img_tensor = F.pad(img_tensor, (pad_W // 2 + pad_W % 2, pad_W // 2, pad_H // 2 + pad_H % 2, pad_H // 2))
        count_mask = torch.ones([1, H_padded, W_padded], device=img_tensor.device)
        image_patches = img_tensor.unfold(2, patch_size, stride_H).unfold(3, patch_size, stride_W).squeeze()
        count_mask = count_mask.unfold(1, patch_size, stride_H).unfold(2, patch_size, stride_W).squeeze()
        image_patches = image_patches.contiguous().view(C, -1, patch_size, patch_size)
        image_patches = image_patches.permute(1, 0, 2, 3)
        count_mask = count_mask.contiguous().view(-1, patch_size, patch_size)

        patch_count.append(image_patches.size(0))
        count_mask_tensors.append(count_mask)
        splitted_tensors.append(image_patches)

    return torch.cat(splitted_tensors, dim=0), count_mask_tensors, patch_count, resized_size


class MatSpectNet(pl.LightningModule):
    """
    Do the RGB2HSI with noise-free RGB as the regularisation term.
    """

    def __init__(self, rgb2hsi: str, hsi2rgb: str, hsiseg: str, optimizer_dict: dict, scheduler_dict: dict,
                 rgb_camera_qe_path,
                 rgb2hsi_checkpoint_path,
                 rgb2hsi_epoch=None,
                 tune_rgb2hsi=False,
                 extrat_features=False,
                 lambda_rgb2hsi=10,
                 lambda_hsi2rgb=10,
                 lambda_idt=0.5,
                 pool_size=30,
                 spectral_filter_number=8,
                 hsiseg_backbone_checkpoint=None,
                 backbone_model="swinv2",
                 image_size=512,
                 segment_classes=16,
                 is_sparse=False,
                 ignore_index=255,
                 lambda_segment=10,
                 use_domain_dis=True,
                 use_materialdb=False,
                 medialog_prefix="testset",
                 use_trainable_rgb_filters=True
                 ):
        super().__init__()
        self.save_hyperparameters()  # log hyper parameters to wandb
        self.rgb2hsi_epoch = rgb2hsi_epoch
        self.automatic_optimization = False
        self.tune_rgb2hsi = tune_rgb2hsi
        self.rgb2hsi_trainable = True
        self.rgb2hsi = spectral_recovery_models.get_models(rgb2hsi, pretrained_model_path=rgb2hsi_checkpoint_path)
        self.rgb2hsi.extrat_features = extrat_features
       
        self.hsiseg = segmentation_models.get_models(model_name=hsiseg, spectral_filter_number=spectral_filter_number,
                                                     backbone_checkpoint=hsiseg_backbone_checkpoint,
                                                     image_size=image_size,
                                                     segment_classes=segment_classes, backbone_model=backbone_model, use_materialdb=use_materialdb)


        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict

        self.criterionCycle = torch.nn.MSELoss()
        self.criterionHSICycle = Loss_MRAE()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionHSIIdt = Loss_MRAE()
        self.criterionSegment = losses.FocalLoss("multiclass", ignore_index=ignore_index, gamma=3)
        self.confmat_valid = MulticlassConfusionMatrix(num_classes=segment_classes, ignore_index=ignore_index)
        self.confmat_test = MulticlassConfusionMatrix(num_classes=segment_classes, ignore_index=ignore_index)
        self.segment_evaluator = SegmentEvaluator(is_sparse=is_sparse, categories=LocalMatDataset.CLASSES)
        self.lambda_rgb2hsi = lambda_rgb2hsi
        self.lambda_hsi2rgb = lambda_hsi2rgb
        self.lambda_idt = lambda_idt
        self.lambda_segment = lambda_segment
        self.criterionGAN = GANLoss("lsgan")

        # scheduler by iteration or epoch
        self.epoch_schedulers = []
        self.step_schedulers = []
        self.stage_schedulers = []

        self.mean_test_importance = []

        self.medialog_prefix = medialog_prefix

    def material_segmentation_forward(self, hsi, rgb):
        pred_mask, filters = self.hsiseg(hsi, rgb)
        return pred_mask, filters

    def backward_segmentor(self, pred_mask, mask_target):
        loss_segmentation = self.criterionSegment(pred_mask, mask_target) * self.lambda_segment
        self.manual_backward(loss_segmentation)
        return loss_segmentation

    def on_train_start(self):
        if self.rgb2hsi_epoch is None:
            self.rgb2hsi_epoch = self.trainer.max_epochs
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]

        for scheduler in schedulers:
            if getattr(scheduler, "by_iteration", False):
                self.step_schedulers.append(scheduler)
            elif getattr(scheduler, "by_stage", False):
                self.stage_schedulers.append(scheduler)
            else:
                self.epoch_schedulers.append(scheduler)

    def training_step(self, batch, batch_idx):
       raise NotImplementedError("not ready for release.")

    def on_train_epoch_end(self):
        raise NotImplementedError("not ready for release.")

    def lr_scheduler_step(self, scheduler, optimizer_idx=None, metric=None):
        if metric is None:
            scheduler.step(None)
        else:
            scheduler.step(metric)

    def training_step_end(self, outputs):
        if self.trainer.current_epoch < self.rgb2hsi_epoch:
            self.log("loss_rgb_recovery", outputs["loss_rgb_recovery"].detach().mean(), prog_bar=False)
            self.log("loss_hsi_recovery", outputs["loss_hsi_recovery"].detach().mean(), prog_bar=False)
            self.log("loss_rgb2hsi", outputs["loss_rgb2hsi"].detach().mean(), prog_bar=False)
            self.log("loss_hsi2rgb", outputs["loss_hsi2rgb"].detach().mean(), prog_bar=False)
            self.log("loss_hsi_discriminator", outputs["loss_hsi_discriminator"].detach().mean(), prog_bar=False)
            self.log("loss_rgb_discriminator", outputs["loss_rgb_discriminator"].detach().mean(), prog_bar=False)
            self.log("l_trans", outputs["l_trans"].detach().mean(), prog_bar=True)
        else:
            self.log("train_loss_segmentation", outputs["train_loss_segmentation"].detach().mean(), prog_bar=True)
        for scheduler in self.step_schedulers:
            self.lr_scheduler_step(scheduler)
        for scheduler in self.stage_schedulers:
            if scheduler.stage == 0:
                self.lr_scheduler_step(scheduler, metric=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop, only LMD is used
        if self.trainer.current_epoch >= self.rgb2hsi_epoch:
            rgb_target, mask_target, _ = batch
            splitted_tensors, count_mask_tensors, patch_count, resized_size = split_overlap_img_tensor(rgb_target)

            pred_hsi, _ = self.rgb2hsi(splitted_tensors)
            pred_mask, filters = self.hsiseg(pred_hsi, splitted_tensors)

            pred_mask = patch2global(pred_mask, count_mask_tensors, patch_count, resized_size)[0]
            valid_loss = self.criterionSegment(pred_mask, mask_target)

            return {"pred_mask": pred_mask, "mask_target": mask_target, "valid_loss": valid_loss}
        else:
            return None

    def validation_step_end(self, outputs):
        # print(outputs["pred_mask"].shape)
        if self.trainer.current_epoch >= self.rgb2hsi_epoch:
            self.confmat_valid.update(outputs["pred_mask"], outputs["mask_target"])
            self.log("valid_segment_loss", outputs["valid_loss"], prog_bar=True)
        else:
            self.log("valid_segment_loss", 999999.99, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.trainer.current_epoch >= self.rgb2hsi_epoch:
            confmat_valid = self.confmat_valid.compute()
            self.confmat_valid.reset()
            _ = self.segment_evaluator(confmat_valid, log_func=self.log, pre_fix="ep_valid")
        else:
            self.log("ep_valid_acc", 0.001, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        rgb_target, mask_target, image_conf = batch
        splitted_tensors, count_mask_tensors, patch_count, resized_size = split_overlap_img_tensor(rgb_target)

        pred_hsi, _ = self.rgb2hsi(splitted_tensors)
        
        pred_mask, filters = self.hsiseg(pred_hsi, splitted_tensors)
        pred_mask = patch2global(pred_mask, count_mask_tensors, patch_count, resized_size)[0]
        test_loss = self.criterionSegment(pred_mask, mask_target)

        return {"pred_mask": pred_mask, "mask_target": mask_target, "test_loss": test_loss}

    def test_step_end(self, outputs):
        self.confmat_test.update(outputs["pred_mask"], outputs["mask_target"])

    def on_test_epoch_end(self):
        confmat_test = self.confmat_test.compute()
        self.confmat_test.reset()
        _ = self.segment_evaluator(confmat_test, log_func=self.log, pre_fix="ep_test")

    def predict_step(self, batch, batch_idx):
        import matplotlib.pyplot as plt
        # this is the pred loop
        rgb_target, mask_target, image_conf = batch
        splitted_tensors, count_mask_tensors, patch_count, resized_size = split_overlap_img_tensor(rgb_target)

        pred_hsi, _ = self.rgb2hsi(splitted_tensors)
        pred_hsi_full = patch2global(pred_hsi, count_mask_tensors, patch_count, resized_size)[0].squeeze().cpu().numpy()

        pred_mask, filters = self.hsiseg(pred_hsi, splitted_tensors)

        pred_mask = patch2global(pred_mask, count_mask_tensors, patch_count, resized_size)[0].argmax(1).squeeze().detach().clone().cpu().numpy()
        mask_target = mask_target.squeeze().detach().clone().cpu().numpy()
        mask_target[mask_target==255] = 16
        # draw rgb detection and upload to log
        pred_mask_rgb = lmd_labels_to_color(pred_mask)
        mask_target_rgb = lmd_labels_to_color(mask_target)

        from PIL import Image
        im = Image.fromarray(pred_mask_rgb)
        im.save("mask.png")

        im = Image.fromarray(pred_hsi_full[0], "L")
        im.save("hsi.png")

    def configure_optimizers(self):
        raise NotImplementedError("not ready for release.")
