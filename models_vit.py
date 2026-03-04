# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import timm.models.vision_transformer


# =========================================================================
# 轻量级空间显著性注意力 (Spatial Saliency Attention)
# 直接从视觉特征中学习 patch 重要性权重，无需跨模态对齐
# 参数量：dim_v×64 + 64 + 64×1 + 1 ≈ 66K (ViT-Large, dim_v=1024)
# =========================================================================
class SpatialSaliencyAttention(nn.Module):
    def __init__(self, dim_v=1024, dim_inner=64):
        super().__init__()

        # 稳住输入范数，保护后续 Linear 的梯度幅度
        self.norm = nn.LayerNorm(dim_v)

        # 两层 MLP：将每个 patch 的特征压缩为一个重要性分数
        self.attn = nn.Sequential(
            nn.Linear(dim_v, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, 1),
        )

        # 零初始化残差门控
        # 训练初期 gamma=0，模块输出为 0，整体退化为纯 baseline
        # 随训练推进自然学习最优贡献权重
        # 随训练推进 gamma 自然向负方向增大
        self.gamma = nn.Parameter(torch.ones(1) * -0.1)

    def forward(self, x_patch):
        # x_patch: [B, N, dim_v]，N=196 (ViT-Large, 224px, patch16)

        # 1. LayerNorm 稳住特征分布
        x_normed = self.norm(x_patch)

        # 2. MLP 预测每个 patch 的重要性分数
        scores = self.attn(x_normed)              # [B, N, 1]

        # 3. Sigmoid 映射到 [0,1]，作为空间显著性掩码
        saliency_mask = torch.sigmoid(scores)     # [B, N, 1]

        # 4. 空间滤波：重要 patch 保留，背景 patch 被抑制
        # [B, N, dim_v] * [B, N, 1] -> [B, N, dim_v]
        guided_feat = (x_patch * saliency_mask).mean(dim=1)  # [B, dim_v]

        # 5. 零初始化残差门控输出
        return guided_feat * self.gamma


# =========================================================================
# 梯度反转层（用于域对抗训练，跨数据集泛化）
# =========================================================================
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


# =========================================================================
# VisionTransformer：挂载 SpatialSaliencyAttention
# =========================================================================
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling """

    def __init__(self, global_pool=False, grad_reverse=0,
                 num_classes=0, num_subjects=0, **kwargs):

        # 传入 num_classes=0 压制父类创建冗余的 1000 维分类头
        super(VisionTransformer, self).__init__(num_classes=0, **kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim  = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            # global_pool 模式不使用 cls_token 路径的 norm
            del self.norm

            # 挂载轻量级空间显著性注意力模块
            self.spatial_attn = SpatialSaliencyAttention(dim_v=embed_dim)

        # 情绪分类头（语义明确的自定义命名，区别于父类的通用 self.head）
        self.AU_head = nn.Linear(kwargs['embed_dim'], num_classes)

        self.grad_reverse = grad_reverse
        print(f"using ID adversarial: {self.grad_reverse}")
        print(f"num classes: {num_classes}, num subjects: {num_subjects}")
        if self.grad_reverse != 0:
            self.ID_head = nn.Linear(kwargs['embed_dim'], num_subjects)
            print(f"activate ID adver head")

    def forward_features(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x)

        # 2. 位置编码（适配 timm >= 0.9，自动处理 cls_token 拼接与 pos_drop）
        x = self._pos_embed(x)

        # 3. 补齐 timm 标准流程中的 norm_pre（标准 ViT 下为 Identity，零开销）
        x = self.norm_pre(x)

        # 4. 经过所有 Transformer Block
        for blk in self.blocks:
            x = blk(x)

        # 5. 特征融合
        if self.global_pool:
            # 去掉 prefix tokens（通常是 1 个 cls_token），仅保留空间 patch
            num_prefix = getattr(self, 'num_prefix_tokens', 1)
            x_patch = x[:, num_prefix:, :]           # [B, 196, embed_dim]

            # 标准全局平均池化（基线特征）
            global_feat = x_patch.mean(dim=1)         # [B, embed_dim]

            # ── 消融开关 ──────────────────────────────────────────
            # 【Baseline】注释下两行，启用这一行：
            # outcome = self.fc_norm(global_feat)

            # 【完整模型】注释上一行，启用下两行：
            spatial_feat = self.spatial_attn(x_patch)         # [B, embed_dim]
            outcome = self.fc_norm(global_feat + spatial_feat)
            # ──────────────────────────────────────────────────────

        else:
            # cls_token 路径（global_pool=False 时保持原始 ViT 行为）
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        AU_pred = self.AU_head(x)

        if self.grad_reverse != 0:
            x = GradReverse.apply(x, self.grad_reverse)
            ID_pred = self.ID_head(x)
            return AU_pred, ID_pred
        return AU_pred


# =========================================================================
# 工厂函数
# =========================================================================

def vit_small_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_base_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_large_patch16(**kwargs):
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def vit_huge_patch14(**kwargs):
    return VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)