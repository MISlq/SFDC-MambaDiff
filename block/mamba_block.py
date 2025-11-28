import torch
from einops import rearrange
from torch import nn
from timm.models.vision_transformer import Attention, Mlp
from block.mamba import Mamba as ssm
# from block.mamba2 import Mamba2 as ssm2
from mamba_ssm import Mamba  # 假设你使用 Mamba 实现
import matplotlib.pyplot as plt
import os
import time

# 创建保存目录
save_dir = "feature_maps"
os.makedirs(save_dir, exist_ok=True)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 spiral mamba block                            #
#################################################################################
class MambaBlock(nn.Module):  # ours
    def __init__(
            self,
            D_dim: int,
            E_dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
            token_list: list,
            token_list_reversal: list,
            origina_list: list,
            origina_list_reversal: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.token_list = token_list
        self.token_list_reversal = token_list_reversal
        self.origina_list = origina_list
        self.origina_list_reversal = origina_list_reversal

        self.norm1 = nn.LayerNorm(D_dim)
        self.mamba1 = ssm(
            d_model=D_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
            token_list=self.token_list,
            token_list_reversal=self.token_list_reversal,
            origina_list=self.origina_list,
            origina_list_reversal=self.origina_list_reversal,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, D_dim * 3, bias=True),
        )

        self.normm = nn.Sequential(
            nn.LayerNorm(2 * D_dim),
            nn.Linear(2 * D_dim, D_dim, bias=True),
            nn.SiLU(),
            nn.Linear(D_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        self.corssatention = CrossAttention(D_dim, D_dim, 0.1)
        self.attn = Attention(D_dim, num_heads=8, qkv_bias=True)
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

    def forward(self, x: torch.Tensor, c1: torch.Tensor, c2:torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c1).chunk(3, dim=1)
        shift_msa2, scale_msa2, gate_msa2 = self.adaLN_modulation(c2).chunk(3, dim=1)
        b, s, d = x.shape
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)

        w_ssm = self.norm1(x)
        w_ssm = modulate(w_ssm, shift_msa2, scale_msa2)

        x_ssm = self.attn(x_ssm)
        w_ssm = self.attn(w_ssm)
        combined_ssm = torch.cat([x_ssm, w_ssm], dim=-1)
        combined_ssm = self.mamba1(combined_ssm, 'z')
        attention_weights = self.normm(combined_ssm)
        x_ssm = attention_weights * x_ssm + (1 - attention_weights) * w_ssm
        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.constant_(self.normm[1].weight, 0)
        nn.init.constant_(self.normm[1].bias, 0)
        nn.init.constant_(self.normm[3].weight, 0)
        nn.init.constant_(self.normm[3].bias, 0)


class MambaBlock_s(nn.Module):  # ours
    def __init__(
            self,
            D_dim: int,
            E_dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
            token_list: list,
            token_list_reversal: list,
            origina_list: list,
            origina_list_reversal: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.token_list = token_list
        self.token_list_reversal = token_list_reversal
        self.origina_list = origina_list
        self.origina_list_reversal = origina_list_reversal

        self.norm1 = nn.LayerNorm(D_dim)

        self.mamba = ssm(
            d_model=D_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
            token_list=self.token_list,
            token_list_reversal=self.token_list_reversal,
            origina_list=self.origina_list,
            origina_list_reversal=self.origina_list_reversal,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, D_dim * 3, bias=True),
        )

        self.attention_network = nn.Sequential(
            nn.LayerNorm(2 * D_dim),
            nn.Linear(2 * D_dim, D_dim, bias=True),
            nn.SiLU(),
            nn.Linear(D_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        self.corssatention = CrossAttention(D_dim, D_dim, 0.1)
        self.attn = Attention(D_dim, num_heads=8, qkv_bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)
        x_ssm = self.mamba(x_ssm, 's')
        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x

class MambaBlock_z(nn.Module):
    def __init__(
            self,
            D_dim: int,
            E_dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
            token_list: list,
            origina_list: list,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.token_list = token_list
        self.origina_list = origina_list
        self.norm1 = nn.LayerNorm(D_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )
        self.mamba = ssm(
            d_model=D_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
            token_list=self.token_list,
            origina_list=self.origina_list,
        )

        self.initialize_weights()
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)
        x_ssm = self.mamba(x_ssm, 'z')
        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x

    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

class MambaBlock_v(nn.Module):
    def __init__(
            self,
            D_dim: int,
            E_dim: int,
            dt_rank: int,
            dim_inner: int,
            d_state: int,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.mamba = ssm(
            d_model=D_dim,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )

        self.norm1 = nn.LayerNorm(D_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim * 2, 3 * D_dim, bias=True),
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        b, s, d = x.shape
        x_ssm = self.norm1(x)
        x_ssm = modulate(x_ssm, shift_msa, scale_msa)
        x_ssm = self.mamba(x_ssm, 'vim')
        x = x + gate_msa.unsqueeze(1) * x_ssm
        return x

    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.attention(query, key, value, attn_mask=mask)
        attn_output = self.dropout(attn_output)
        output = self.norm(query + attn_output)
        return output
