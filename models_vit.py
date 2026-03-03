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
        
        # 稳住输入范数
        self.input_norm = nn.LayerNorm(dim_v)
        
        # 视觉侧降维投影
        self.q_proj = nn.Linear(dim_v, dim_inner)
        
        # 缩放因子：1 / sqrt(128) ≈ 0.088
        self.scale = dim_inner ** -0.5
        
        # =======================================================
        # 防御性加载逻辑与 Tensor 断言
        # =======================================================
        text_features = torch.load(text_features_path, map_location='cpu', weights_only=False)
        if isinstance(text_features, dict):
            keys = list(text_features.keys())
            text_features = text_features[keys[0]]
            
        assert isinstance(text_features, torch.Tensor), \
            f"期望 Tensor，实际得到 {type(text_features)}"
            
        text_features = text_features[list(mapping_idx)]
        text_features = text_features.float()
        
        # =======================================================
        # 文本侧神来之笔：DDP 安全的静态正交降维 (4096 -> 128)
        # 固定随机种子，防止多卡训练时各进程目标坐标系不一致导致梯度对冲
        # =======================================================
        with torch.no_grad():
            generator = torch.Generator()
            generator.manual_seed(42)  # 固定种子保证所有 GPU 生成同一套坐标系
            
            # 利用 QR 分解提取天然正交矩阵 (Q 矩阵)
            random_matrix = torch.randn(dim_t, dim_inner, generator=generator)
            random_proj = torch.linalg.qr(random_matrix)[0]
            
            # 将文本特征降维并确立标准的 128 维方向系
            text_keys_128 = text_features @ random_proj
            text_keys_128 = F.normalize(text_keys_128, p=2, dim=-1)
            
        self.register_buffer('text_keys', text_keys_128)  # [7, 128]
        
        # 0.1 初始动量
        # sigmoid(-2.197) ≈ 0.1，保持初始权重与之前一致
        self.gamma = nn.Parameter(torch.ones(1) * -2.197)

    def forward(self, x_patch):
        # AMP 豁免区：强制在 FP32 下运行
        device_type = x_patch.device.type if x_patch.device.type in ['cuda', 'cpu'] else 'cuda'
        with torch.autocast(device_type=device_type, enabled=False):
            x_fp32 = x_patch.float()
            x_normed = self.input_norm(x_fp32)
            
            # Q 的维度现在是 [B, 196, 128]
            q = self.q_proj(x_normed)  
            
            # Scaled Dot-Product Attention
            attn_scores = (q @ self.text_keys.T) * self.scale  # [B, 196, 7]
            
        # 后续操作对精度不敏感
        max_scores, _ = torch.max(attn_scores, dim=-1)
        saliency_mask = torch.sigmoid(max_scores)
        
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            num_prefix = getattr(self, 'num_prefix_tokens', 1)
            x_patch = x[:, num_prefix:, :]  
            
            global_feat = x_patch.mean(dim=1)
            text_guided_feat = self.text_attn(x_patch) 
            
            # 融合后进行 LayerNorm
            outcome = self.fc_norm(global_feat + text_guided_feat)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
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