import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
import torch.nn.functional as F

from itertools import repeat
import collections.abc
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained('pretrain').to(device)

class CFP_Encoder(nn.Module):
    def __init__(
        self,
        img_size=64,  # 28,
        patch_size=2,
        in_channels=4,
        embed_dim=512,
        contain_mask_token=True,
        reduction_ratio=14,
        hidden_size=512,
        strip_size=2,
        depth=16,
        dt_rank=16,
        d_state=16,
    ):
        super().__init__()
        self.depth = depth
        self.vision_embedding = VisionEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            contain_mask_token=contain_mask_token, 
        )

        self.PyramidConv = PyramidConv()
        self.PyramidAdaptiveAvgPool = PyramidAdaptiveAvgPool()

        self.x_embedder = PatchEmbed(img_size, patch_size, strip_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.attention_network1 = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2* embed_dim, 512, bias=True),

        )

        self.blocks_z = nn.ModuleList([
            MambaBlock_z(token_list=z(int(img_size / patch_size), i)[0],
                           origina_list=z(int(img_size / patch_size), i)[1],
                           D_dim=hidden_size,
                           E_dim=hidden_size * 2,
                           dim_inner=hidden_size * 2,
                           dt_rank=dt_rank,
                           d_state=d_state,)
            for i in range(depth)
        ])

        self.blocks_v = nn.ModuleList([
            MambaBlock_v(D_dim=hidden_size,
                           E_dim=hidden_size * 2,
                           dim_inner=hidden_size * 2,
                           dt_rank=dt_rank,
                           d_state=d_state, )
            for j in range(depth)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x,save_dir='32/'):
        max_out = self.PyramidConv(x, save_dir=save_dir)
        avg_out = self.PyramidAdaptiveAvgPool(x, save_dir=save_dir)

        max_out_x = self.x_embedder(max_out) + self.pos_embed
        avg_out_x =  self.x_embedder(avg_out) + self.pos_embed

        block_max_outputs = []
        for i in range(self.depth):
            if i == 0:
                max_out_x = self.blocks_z[i](max_out_x)  # w
            elif i > self.depth / 2:
                skip_connection = block_max_outputs[self.depth - i - 1]
                max_out_x = self.blocks_z[i](block_max_outputs[-1] + skip_connection)  # w
            else:
                max_out_x = self.blocks_z[i](block_max_outputs[-1])
            block_max_outputs.append(max_out_x)

        block_avg_outputs = []
        for j in range(self.depth):
            if j == 0:
                avg_out_x = self.blocks_v[j](avg_out_x)  # w
            elif j > self.depth / 2:
                skip_connection = block_avg_outputs[self.depth - j - 1]
                avg_out_x = self.blocks_v[j](block_avg_outputs[-1] + skip_connection)  # w
            else:
                avg_out_x = self.blocks_v[j](block_avg_outputs[-1])
            block_avg_outputs.append(avg_out_x)

        combined_ssm = torch.cat([max_out_x, avg_out_x], dim=-1)        #(8,1024,1024)
        out = self.attention_network1(combined_ssm)           #(8,1024,512)
        return out

class PatchEmbed(nn.Module):

    def __init__(self, img_size=28, patch_size=2, stride=2, in_chans=4, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
class PyramidAdaptiveAvgPool(nn.Module):
    def __init__(self, output_size=(64, 64)):
        super(PyramidAdaptiveAvgPool, self).__init__()
        self.output_size = output_size
        self.pool_3x3 = nn.AdaptiveAvgPool2d(3)
        self.pool_5x5 = nn.AdaptiveAvgPool2d(5)
        self.pool_7x7 = nn.AdaptiveAvgPool2d(7)
        self.conv1x1 = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1)

    def forward(self, x):
        target_size = self.output_size  # e.g. (64, 64)

        x_3 = self.pool_3x3(x)
        x_5 = self.pool_5x5(x)
        x_7 = self.pool_7x7(x)

        x_3_up = F.interpolate(x_3, size=target_size, mode='bilinear', align_corners=False)
        x_5_up = F.interpolate(x_5, size=target_size, mode='bilinear', align_corners=False)
        x_7_up = F.interpolate(x_7, size=target_size, mode='bilinear', align_corners=False)

        out = torch.cat([x_3_up, x_5_up, x_7_up], dim=1)  # dim=1 为通道维度
        out = self.conv1x1(out)
        return out

class PyramidConv(nn.Module):
    def __init__(self):
        super(PyramidConv, self).__init__()
        self.conv_3x3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=7, padding=3)
        self.conv1x1 = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1)

    def forward(self, x):
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_7x7 = self.conv_7x7(x)

        out = torch.cat((x_3x3, x_5x5, x_7x7), dim=1)
        out = self.conv1x1(out)
        return out
