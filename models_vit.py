# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import timm.models.vision_transformer

# =========================================================================
# 前沿架构创新：文本引导显著性掩码 (Text-Guided Saliency Mask) - 低维正交投影版
# 极低参数量 (131K)，内置 DDP 多卡统一随机坐标系防对冲设计
# =========================================================================
class TextGuidedSaliencyMask(nn.Module):
    def __init__(self, dim_v=1024, dim_t=4096, dim_inner=128,
                 text_features_path='/Data/hjt/affectnet_7class_au_text_features.pt', 
                 mapping_idx=(3, 4, 5, 1, 2, 6, 0)):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(dim_v)
        self.q_proj = nn.Linear(dim_v, dim_inner)
        self.scale = dim_inner ** -0.5
        
        text_features = torch.load(text_features_path, map_location='cpu', weights_only=False)
        if isinstance(text_features, dict):
            keys = list(text_features.keys())
            text_features = text_features[keys[0]]
            
        assert isinstance(text_features, torch.Tensor), \
            f"期望 Tensor，实际得到 {type(text_features)}"
            
        text_features = text_features[list(mapping_idx)]
        text_features = text_features.float()
        
        with torch.no_grad():
            generator = torch.Generator()
            generator.manual_seed(42)
            random_matrix = torch.randn(dim_t, dim_inner, generator=generator)
            random_proj = torch.linalg.qr(random_matrix)[0]
            text_keys_128 = text_features @ random_proj
            text_keys_128 = F.normalize(text_keys_128, p=2, dim=-1)
            
        self.register_buffer('text_keys', text_keys_128)  # [7, 128]
        self.gamma = nn.Parameter(torch.ones(1) * -2.197)

    def forward(self, x_patch, prior_mask=None):
        """
        x_patch    : [B, 196, 1024]
        prior_mask : [B, 196, 1]，AU 几何先验掩码（可选，None 时退化为原始行为）
        """
        device_type = x_patch.device.type if x_patch.device.type in ['cuda', 'cpu'] else 'cuda'
        with torch.autocast(device_type=device_type, enabled=False):
            x_fp32 = x_patch.float()
            x_normed = self.input_norm(x_fp32)
            q = self.q_proj(x_normed)              # [B, 196, 128]
            attn_scores = (q @ self.text_keys.T) * self.scale  # [B, 196, 7]

        max_scores, _ = torch.max(attn_scores, dim=-1)
        saliency_mask = torch.sigmoid(max_scores)  # [B, 196]

        # ── AU 先验约束：学习掩码 × 几何先验掩码 ────────────────────────────
        if prior_mask is not None:
            # prior_mask: [B, 196, 1] → [B, 196]
            prior = prior_mask.squeeze(-1).to(saliency_mask.dtype)
            saliency_mask = saliency_mask * prior
        # ────────────────────────────────────────────────────────────────────

        guided_patch = x_patch * saliency_mask.unsqueeze(-1).to(x_patch.dtype)
        guided_feat = guided_patch.mean(dim=1)

        return guided_feat * self.gamma.clamp(max=1.0)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, grad_reverse=0, num_classes=0, num_subjects=0, 
                 text_features_path='/Data/hjt/affectnet_7class_au_text_features.pt',
                 mapping_idx=(3, 4, 5, 1, 2, 6, 0),
                 dim_t=4096, **kwargs):
        
        super(VisionTransformer, self).__init__(num_classes=0, **kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  
            
            self.text_attn = TextGuidedSaliencyMask(
                dim_v=embed_dim, dim_t=dim_t,
                text_features_path=text_features_path,
                mapping_idx=mapping_idx
            )

        self.AU_head = nn.Linear(kwargs['embed_dim'], num_classes)
        self.grad_reverse = grad_reverse 
        if not self.grad_reverse == 0:
            self.ID_head = nn.Linear(kwargs['embed_dim'], num_subjects)

    def forward_features(self, x, prior_mask=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            num_prefix = getattr(self, 'num_prefix_tokens', 1)
            x_patch = x[:, num_prefix:, :]

            global_feat = x_patch.mean(dim=1)
            # ── prior_mask 传入 text_attn ────────────────────────────────────
            text_guided_feat = self.text_attn(x_patch, prior_mask=prior_mask)
            # ────────────────────────────────────────────────────────────────
            outcome = self.fc_norm(global_feat + text_guided_feat)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, prior_mask=None):
        x = self.forward_features(x, prior_mask=prior_mask)
        AU_pred = self.AU_head(x)

        if not self.grad_reverse == 0:
            x = GradReverse.apply(x, self.grad_reverse)
            ID_pred = self.ID_head(x)
            return AU_pred, ID_pred
        else:
            return AU_pred


def vit_small_patch16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_base_patch16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_large_patch16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def vit_huge_patch14(**kwargs):
    return VisionTransformer(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)