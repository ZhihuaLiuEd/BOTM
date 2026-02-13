import math
from functools import partial
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from einops import rearrange
from einops.layers.torch import Rearrange


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for 2D images (B, C, H, W)"""
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels * expansion, kernel_size=3, groups=channels, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, kernel_size=1)
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

class TransportPlanProduct(nn.Module):
    """
    Computes the Optimal Transport plan and applies it to Value.
    Implements the core Sinkhorn-like logic or simplified OT attention.
    """
    def __init__(self, temperature, attn_dropout=0.1, eps=0.5):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.eps = eps

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self, q, k, v, mask=None):
        # q, k, v: [B, Heads, HW, C]
        
        # 1. Similarity Measurement for Cost Matrix approximation
        k_l2 = self._l2norm(k, dim=-1)
        v_l2 = self._l2norm(v, dim=-1)
        kv_sim = torch.clamp(k_l2 * v_l2, min=0)
        
        # 2. Approximate Transport Plan
        diff = 1 - kv_sim
        K_mat = torch.exp(-diff / self.eps) # [B, Heads, HW, C]

        # 3. Sinkhorn-like Iterations (Simplified for Efficiency)
        B, heads, HW, channel = K_mat.shape
        
        # Random initialization for vectors a and b
        a = torch.randn(diff.shape, device=q.device) * math.sqrt(2. / channel)
        a = self._l2norm(a, dim=1)

        # Iterative update (fixed iterations for speed)
        b = k / K_mat * a
        a = v / K_mat * b
        k_updated = k * a

        # 4. Attention Map Calculation
        attn_OT = torch.matmul(q / self.temperature, k_updated.transpose(2, 3))
        
        if mask is not None:
            attn_OT = attn_OT.masked_fill(mask == 0, -1e9)
            
        attn_OT = self.dropout(F.softmax(attn_OT, dim=-1))
        
        # 5. Output
        output = torch.matmul(attn_OT, v)
        
        return output, attn_OT

class OTAttention(nn.Module):
    """
    Optimal Transport Attention Module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., drop_out=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.w_qs = nn.Linear(dim, dim * num_heads, bias=qkv_bias)
        self.w_ks = nn.Linear(dim, dim * num_heads, bias=qkv_bias)
        self.w_vs = nn.Linear(dim, dim * num_heads, bias=qkv_bias)
        self.fc = nn.Linear(dim * num_heads, dim, bias=qkv_bias)

        self.attention = TransportPlanProduct(temperature=dim ** 0.5)

        self.drop_out = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x_q, x_k, x_v, mask=None):
        B, HW, C = x_q.shape
        residual = x_q

        # Linear Projections
        q = self.w_qs(x_q).view(B, -1, self.num_heads, self.dim).transpose(1, 2)
        k = self.w_ks(x_k).view(B, -1, self.num_heads, self.dim).transpose(1, 2)
        v = self.w_vs(x_v).view(B, -1, self.num_heads, self.dim).transpose(1, 2)

        # Attention
        if mask is not None:
            mask = mask.unsqueeze(-1)
        
        q_attn, _ = self.attention(q, k, v, mask=mask)
        
        # Projection and Residual
        q_attn = q_attn.transpose(1, 2).contiguous().view(B, HW, -1)
        q_attn = self.drop_out(self.fc(q_attn))
        
        # Post-Norm
        q_attn = self.layer_norm(q_attn + residual)
        return q_attn

class BOTMBlock(nn.Module):
    """
    Barycentric Optimal Transport Matching Block.
    Computes Cross-Attention between two frames (ES and ED).
    """
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio),
            LayerNorm2d(channels),
        )
        self.att = OTAttention(channels, num_heads=num_heads)

    def forward(self, x_ES, x_ED):
        # x_ES, x_ED: [B, C, H, W]
        _, _, h, w = x_ES.shape
        
        # Reduce spatial dimensions for Key and Value
        reduced_x_ES = self.reducer(x_ES)
        reduced_x_ED = self.reducer(x_ED)

        # Flatten: [B, C, H, W] -> [B, HW, C]
        flat_x_ES = rearrange(x_ES, "b c h w -> b (h w) c")
        flat_reduced_x_ES = rearrange(reduced_x_ES, "b c h w -> b (h w) c")
        
        flat_x_ED = rearrange(x_ED, "b c h w -> b (h w) c")
        flat_reduced_x_ED = rearrange(reduced_x_ED, "b c h w -> b (h w) c")

        BOTM_ES_OTA = self.att(flat_x_ES, flat_reduced_x_ED, flat_reduced_x_ES)
        BOTM_ES_OTA = rearrange(BOTM_ES_OTA, "b (h w) c -> b c h w", h=h, w=w)

        BOTM_ED_OTA = self.att(flat_x_ED, flat_reduced_x_ES, flat_reduced_x_ED)
        BOTM_ED_OTA = rearrange(BOTM_ED_OTA, "b (h w) c -> b c h w", h=h, w=w)

        return BOTM_ES_OTA, BOTM_ED_OTA

class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio),
            LayerNorm2d(channels)
        )
        self.att = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        out, _ = self.att(x_flat, reduced_x, reduced_x)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

class SegFormerEncoderBlock(nn.Sequential):
    def __init__(self, channels, reduction_ratio=1, num_heads=8, mlp_expansion=4, drop_path_prob=0.0):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            )
        )

class OverlapPatchMerging(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size, overlap_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=overlap_size, 
                      padding=patch_size//2, bias=False),
            LayerNorm2d(out_channels)
        )

class SegFormerEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, overlap_size, 
                 drop_probs, depth, reduction_ratio, num_heads, mlp_expansion):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size
        )
        self.blocks = nn.Sequential(*[
            SegFormerEncoderBlock(out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]) 
            for i in range(depth)
        ])
        self.norm = LayerNorm2d(out_channels)

    def forward(self, x):
        x = self.overlap_patch_merge(x)
        x = self.blocks(x)
        return self.norm(x)

class SegFormerEncoder(nn.Module):
    def __init__(self, in_channels, widths, depths, all_num_heads, patch_sizes, 
                 overlap_sizes, reduction_ratios, mlp_expansions, drop_prob=0.0,
                 # BOTM specific params
                 reduction_botm=[4, 2, 1, 1], all_num_heads_botm=[1, 2, 4, 8]):
        super().__init__()
        
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        
        self.stages = nn.ModuleList()
        self.botm_stages = nn.ModuleList()
        
        curr_in_channels = in_channels
        curr_drop_idx = 0
        
        for i in range(len(widths)):
            stage_drop_probs = drop_probs[curr_drop_idx : curr_drop_idx + depths[i]]
            curr_drop_idx += depths[i]
            
            self.stages.append(SegFormerEncoderStage(
                in_channels=curr_in_channels,
                out_channels=widths[i],
                patch_size=patch_sizes[i],
                overlap_size=overlap_sizes[i],
                drop_probs=stage_drop_probs,
                depth=depths[i],
                reduction_ratio=reduction_ratios[i],
                num_heads=all_num_heads[i],
                mlp_expansion=mlp_expansions[i]
            ))
            
            self.botm_stages.append(BOTMBlock(
                channels=widths[i],
                reduction_ratio=reduction_botm[i],
                num_heads=all_num_heads_botm[i]
            ))
            
            curr_in_channels = widths[i]

    def forward(self, x_ES, x_ED):
        features_ES = []
        features_ED = []
        ES_OT_list = []
        ED_OT_list = []
        
        for stage, botm in zip(self.stages, self.botm_stages):
            # 1. Independent Feature Extraction
            x_ES = stage(x_ES)
            x_ED = stage(x_ED)

            # 2. Optimal Transport Interaction
            x_ES_OT, x_ED_OT = botm(x_ES, x_ED)

            features_ES.append(x_ES)
            features_ED.append(x_ED)
            ES_OT_list.append(x_ES_OT)
            ED_OT_list.append(x_ED_OT)
            
        return features_ES, features_ED, ES_OT_list, ED_OT_list

class SegFormerDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def barycenter_fusion(self, es, ed, es_OT, ed_OT, alpha=0.8):
        """
        Barycenter Fusion: Fuses original features with OT-enhanced features.
        alpha: Weight for the primary OT direction.
        """
        x_ES_fused = alpha * es_OT + (1 - alpha) * ed_OT
        x_ED_fused = (1 - alpha) * es_OT + alpha * ed_OT
        
        return es + x_ES_fused, ed + x_ED_fused

    def forward(self, input_ES, input_ED, input_ES_OT, input_ED_OT):
        x1, x2 = self.barycenter_fusion(input_ES, input_ED, input_ES_OT, input_ED_OT)
        
        x1 = self.conv(self.upsample(x1))
        x2 = self.conv(self.upsample(x2))
        return x1, x2

class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels, widths, scale_factors):
        super().__init__()
        self.stages = nn.ModuleList([
            SegFormerDecoderBlock(in_c, out_channels, scale_f)
            for in_c, scale_f in zip(widths, scale_factors)
        ])

    def forward(self, features_ES, features_ED, es_OT, ed_OT):
        es_decode = []
        ed_decode = []
        for i, stage in enumerate(self.stages):
            es, ed = stage(features_ES[i], features_ED[i], es_OT[i], ed_OT[i])
            es_decode.append(es)
            ed_decode.append(ed)
        return es_decode, ed_decode

class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels, num_classes, num_features=4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels)
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, features_list):
        x = torch.cat(features_list, dim=1)
        x = self.fuse(x)
        x = self.upsample(x)
        x = self.predict(x)
        return x

class SegFormerBOTM(nn.Module):
    """
    Main Model Class.
    Input: Pair of Frames (ES, ED)
    Output: Segmentation Masks for both frames
    """
    def __init__(self, 
                 in_channels=1, 
                 num_classes=4,
                 widths=[64, 128, 256, 512],
                 depths=[3, 4, 6, 3],
                 all_num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 overlap_sizes=[4, 2, 2, 2],
                 reduction_ratios=[8, 4, 2, 1],
                 mlp_expansions=[4, 4, 4, 4],
                 decoder_channels=256,
                 scale_factors=[1, 2, 4, 8],
                 drop_prob=0.0):
        super().__init__()

        self.encoder = SegFormerEncoder(
            in_channels=in_channels,
            widths=widths,
            depths=depths,
            all_num_heads=all_num_heads,
            patch_sizes=patch_sizes,
            overlap_sizes=overlap_sizes,
            reduction_ratios=reduction_ratios,
            mlp_expansions=mlp_expansions,
            drop_prob=drop_prob
        )
        
        self.decoder = SegFormerDecoder(decoder_channels, widths, scale_factors)
        
        self.head_es = SegFormerSegmentationHead(decoder_channels, num_classes, num_features=len(widths))
        self.head_ed = SegFormerSegmentationHead(decoder_channels, num_classes, num_features=len(widths))

    def forward(self, ES_frame, ED_frame):
        ES_feats, ED_feats, ES_OT, ED_OT = self.encoder(ES_frame, ED_frame)
        ES_decoded, ED_decoded = self.decoder(ES_feats, ED_feats, ES_OT, ED_OT)
        
        pred_es = self.head_es(ES_decoded)
        pred_ed = self.head_ed(ED_decoded)
        
        return pred_es, pred_ed

def get_botm_model(config: dict):
    return SegFormerBOTM(**config)

if __name__ == '__main__':
    model = SegFormerBOTM(in_channels=1, num_classes=4, decoder_channels=128)
    x1 = torch.randn(2, 1, 448, 448)
    x2 = torch.randn(2, 1, 448, 448)
    
    y1, y2 = model(x1, x2)
    print(f"Input: {x1.shape}")
    print(f"Output: {y1.shape}, {y2.shape}")