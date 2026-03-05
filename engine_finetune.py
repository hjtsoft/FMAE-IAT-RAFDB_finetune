# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy
from timm.loss import LabelSmoothingCrossEntropy

import util.misc as misc
import util.lr_sched as lr_sched


class TextGuidedSoftLabelLoss(torch.nn.Module):
    def __init__(self, text_features_path, alpha=0.3, tau=0.5, smoothing=0.15):
        super().__init__()
        self.alpha = alpha

        text_features = torch.load(text_features_path, map_location='cpu', weights_only=False)
        if isinstance(text_features, dict):
            text_features = text_features.get('features', text_features)

        affectnet_to_rafdb_idx = [3, 4, 5, 1, 2, 6, 0]
        text_features = text_features[affectnet_to_rafdb_idx]

        text_features = F.normalize(text_features.float(), p=2, dim=-1)
        sim_matrix = text_features @ text_features.T

        soft_labels = F.softmax(sim_matrix / tau, dim=1)
        self.register_buffer('text_soft_labels', soft_labels)

        self.hard_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, outputs, targets):
        hard_loss = self.hard_criterion(outputs, targets)
        batch_soft_targets = self.text_soft_labels[targets]
        log_probs = F.log_softmax(outputs, dim=1)
        soft_loss = -(batch_soft_targets * log_probs).sum(dim=1).mean()
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, prior_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples    = samples.to(device, non_blocking=True)
        targets    = targets.to(device, non_blocking=True)
        prior_mask = prior_mask.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast(device_type='cuda'):
            outputs = model(samples, prior_mask=prior_mask)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        # =======================================================
        # 探针：监控 TextGuided 模块的梯度健康度 (仅在第一个 Batch 打印)
        # =======================================================
        if epoch == 0 and data_iter_step == 0:
            if hasattr(model, 'module'):
                attn_module = model.module.text_attn
            else:
                attn_module = model.text_attn

            q_proj_grad = attn_module.q_proj.weight.grad
            gamma_val = attn_module.gamma.data.item()
            if q_proj_grad is not None:
                print(f"\n[梯度探针] Gamma 初始值: {gamma_val:.4f}")
                print(f"[梯度探针] q_proj.weight 梯度绝对值均值: {q_proj_grad.abs().mean().item():.2e}")
        # =======================================================

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 100)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# =======================================================
# TTA 推理函数
# =======================================================
def tta_inference(model, images, prior_mask, device):
    """
    Test-Time Augmentation：5 种变换取概率平均
    prior_mask 不做空间变换（掩码是 patch 级别，与图像裁剪无关）
    """
    def center_crop_resize(imgs, crop_size):
        h, w = imgs.shape[-2], imgs.shape[-1]
        top  = (h - crop_size) // 2
        left = (w - crop_size) // 2
        cropped = imgs[:, :, top:top+crop_size, left:left+crop_size]
        return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)

    augmented = [
        images,
        images.flip(-1),
        center_crop_resize(images, 208),
        center_crop_resize(images, 208).flip(-1),
        center_crop_resize(images, 192),
    ]

    with torch.autocast(device_type=device.type):
        logits_list = [model(aug, prior_mask=prior_mask) for aug in augmented]

    probs = torch.stack([torch.softmax(l, dim=-1) for l in logits_list], dim=0)
    return probs.mean(dim=0)


@torch.no_grad()
def evaluate(data_loader, model, device, use_tta=False):
    """
    use_tta=False : 标准单次推理（训练中每 epoch 的验证，速度快）
    use_tta=True  : TTA 推理（最终结果汇报时使用，推理时间约 5×）
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test (TTA):' if use_tta else 'Test:'

    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        images     = batch[0].to(device, non_blocking=True)
        target     = batch[1].to(device, non_blocking=True)
        prior_mask = batch[2].to(device, non_blocking=True)

        if use_tta:
            output = tta_inference(model, images, prior_mask, device)
            with torch.autocast(device_type=device.type):
                logits_orig = model(images, prior_mask=prior_mask)
                loss = criterion(logits_orig, target)
        else:
            with torch.autocast(device_type=device.type):
                output = model(images, prior_mask=prior_mask)
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}