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

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy

class TextGuidedSoftLabelLoss(torch.nn.Module):
    def __init__(self, text_features_path, alpha=0.3, tau=0.5, smoothing=0.15):
        """
        alpha: 控制 Llama-3 软标签 Loss 的权重比 (0~1)
        tau: 温度系数，值越大分布越平滑。默认设为 0.5，保留类别间合理差异。
        """
        super().__init__()
        self.alpha = alpha
        
        # 1. 加载 Llama-3 提取的文本特征
        text_features = torch.load(text_features_path, map_location='cpu', weights_only=False)
        if isinstance(text_features, dict):
            text_features = text_features.get('features', text_features)
            
        # =========================================================================
        # [修复问题 1: 对齐 AffectNet 与 RAF-DB 的类别顺序]
        # 假设你当初提取 AffectNet 文本特征时的顺序是标准的：
        #   [0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger]
        # 而 RAF-DB 训练时(经过 -1 处理后)的顺序是：
        #   [0: Surprise, 1: Fear, 2: Disgust, 3: Happiness, 4: Sadness, 5: Anger, 6: Neutral]
        # 
        # 下面列表的意思是：取 AffectNet 的第 3 号、第 4 号... 重新组合成 RAF-DB 的顺序。
        # ⚠️ 请务必回忆/检查你保存 .pt 时用的文本列表顺序，如果不同，请修改此 mapping！
        # =========================================================================
        affectnet_to_rafdb_idx = [3, 4, 5, 1, 2, 6, 0] 
        text_features = text_features[affectnet_to_rafdb_idx]
        
        # 2. L2 归一化并计算类别间的余弦相似度矩阵
        text_features = F.normalize(text_features.float(), p=2, dim=-1)
        sim_matrix = text_features @ text_features.T
        
        # 3. [修复问题 2 & 3: 增大 tau 到 0.5，并注册为 buffer 以规范化计算图和设备挂载]
        soft_labels = F.softmax(sim_matrix / tau, dim=1)
        self.register_buffer('text_soft_labels', soft_labels)
        
        # 保留原有的平滑策略作为基础 Loss
        self.hard_criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(self, outputs, targets):
        # 1. 计算原始的 Hard/Smoothing Loss
        hard_loss = self.hard_criterion(outputs, targets)
        
        # 2. 从 buffer 中找到 Batch 对应的 Llama-3 软标签 [B, 7] (自动与 outputs 同设备)
        batch_soft_targets = self.text_soft_labels[targets]
        
        # 3. 计算预测分布与 LLM 软标签的交叉熵 (软蒸馏)
        log_probs = F.log_softmax(outputs, dim=1)
        soft_loss = -(batch_soft_targets * log_probs).sum(dim=1).mean()
        
        # 4. 加权融合返回
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
    print_freq = 200  # print log every 20 steps
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast(device_type='cuda'):
            outputs = model(samples)
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
            if hasattr(model, 'module'): # 处理 DDP 封装
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
            """ We use epoch_100x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 100)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.autocast(device_type='cuda'):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}