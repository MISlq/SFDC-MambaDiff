
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from loader_train import PatientDataGenerator
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
import wandb
from glob import glob
from loguru import logger
from time import time
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL


class LearnableHaar:
    def __init__(self):
        self.filters = [
            nn.Parameter(torch.tensor([0.5, 0.5])),
            nn.Parameter(torch.tensor([-0.5, 0.5]))
        ]
    def __len__(self):
        return len(self.filters)

    def __getitem__(self, idx):
        """返回张量形式的滤波器"""
        return self.filters[idx].detach().cpu().numpy()

class WaveletModel(torch.nn.Module):
    def __init__(
         self,
         img_size=64,
         patch_size=2,
         hidden_size=512,
         depth=12,
         dt_rank=16,
         d_state=16,
         device='cuda:0'):
        super().__init__()
        self.device = device

        self.dwt = DWTForward(J=2, mode='zero', wave=LearnableHaar()).to(device)
        self.idwt = DWTInverse(mode='zero', wave=LearnableHaar()).to(device)

        self.energy_weights = nn.Parameter(torch.ones(7) / 7)

        self.mamba = Mamba(28)

        self.decoder = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),  # 输入通道数 4（输出图像C=4）
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks_s = nn.ModuleList([
            MambaBlock_s(token_list=s(int(img_size / patch_size), i)[0],
                         origina_list=s(int(img_size / patch_size), i)[1],
                         D_dim=hidden_size,
                         E_dim=hidden_size * 2,
                         dim_inner=hidden_size * 2,
                         dt_rank=dt_rank,
                         d_state=d_state, )
            for i in range(depth)
        ])

    def forward(self, x):

        Yl, Yh = self.dwt(x)
        LL2 = Yl
        LH2 = Yh[1][:, :, 0, :, :]
        HL2 = Yh[1][:, :, 1, :, :]
        HH2 = Yh[1][:, :, 2, :, :]
        LH1 = Yh[0][:, :, 0, :, :]
        HL1 = Yh[0][:, :, 1, :, :]
        HH1 = Yh[0][:, :, 2, :, :]

        bands = [LL2, LH2, HL2, HH2, LH1, HL1, HH1]
        weights = F.softmax(self.energy_weights, dim=0)
        E_total = sum(torch.norm(b, p=2) * w for b, w in zip(bands, weights))
        E_ratio = torch.stack([torch.norm(b, p=2) / E_total for b in bands])
        self.loss_energy = torch.var(E_ratio) * 0.6  # 可调节权重

        target_size = LL2.shape[2:]  # (H, W)
        resize = lambda t: torch.nn.functional.interpolate(t, size=target_size, mode='bilinear', align_corners=False)

        bands = [LL2, LH2, HL2, HH2, LH1, HL1, HH1]
        resized_bands = [resize(b) for b in bands]

        fused = torch.cat(resized_bands, dim=1)

        fused = rearrange(fused, 'b c h w -> b h w c')  # [B, H, W, 7C]
        B, H, W, C = fused.shape
        fused = fused.view(B, H * W, C)

        block_fused_out = []
        for i in range(self.depth):
            if i == 0:
                fused_out = self.blocks_s[i](fused)  # w
            elif i > self.depth / 2:
                skip_connection = block_fused_out[self.depth - i - 1]
                fused_out = self.blocks_s[i](block_fused_out[-1] + skip_connection)  # w
            else:
                fused_out = self.blocks_s[i](block_fused_out[-1])
            block_fused_out.append(fused_out)

        fused_out = fused_out.view(B, H, W, C)
        fused_out = rearrange(fused_out, 'b h w c -> b c h w')

        LL2_, LH2_, HL2_, HH2_, LH1_, HL1_, HH1_ = torch.chunk(fused_out, chunks=7, dim=1)

        def resize_back(x, ref):
            return torch.nn.functional.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

        LL2_ = resize_back(LL2_, LL2)
        LH2_ = resize_back(LH2_, LH2)
        HL2_ = resize_back(HL2_, HL2)
        HH2_ = resize_back(HH2_, HH2)
        LH1_ = resize_back(LH1_, LH1)
        HL1_ = resize_back(HL1_, HL1)
        HH1_ = resize_back(HH1_, HH1)

        Yl_rec = LL2_
        Yh_rec = [
            torch.stack([LH1_, HL1_, HH1_], dim=2),  # 第一级高频子带
            torch.stack([LH2_, HL2_, HH2_], dim=2)  # 第二级高频子带
        ]

        self.loss_grad = self._gradient_loss(Yh_rec, Yh) * 0.05

        out = self.idwt((Yl_rec, Yh_rec))

        out = self.decoder(out)

        out = out.flatten(2).transpose(1, 2)
        return out

    def _gradient_loss(self, Yh_rec, Yh):
        loss = 0
        for rec, orig in zip(Yh_rec, Yh):
            def prepare(t):
                if t.dim() == 4:
                    return t
                elif t.dim() == 5:
                    return t.squeeze(2)
                else:
                    return t
            orig = prepare(orig)
            rec = prepare(rec)

            for dim in [2, 3]:  # 高度和宽度维度
                grad_rec = torch.diff(rec, dim=dim)
                grad_orig = torch.diff(orig, dim=dim)
                loss += F.l1_loss(grad_rec, grad_orig)

        return loss

def create_logger(logging_dir):
    logger.add(f"{logging_dir}/log.txt", format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
    return logger

def infoNCE_loss_b(input_tensor, tau=0.07):
    batch_size, seq_len, feat_dim = input_tensor.shape
    reshaped_tensor = input_tensor.reshape(batch_size, seq_len*feat_dim)
    reshaped_tensor = F.normalize(reshaped_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(reshaped_tensor, reshaped_tensor.T) / tau
    labels = torch.arange(reshaped_tensor.size(0), dtype=torch.long, device=reshaped_tensor.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    seed = args.embedder_global_seed
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(project="dwt_encoder")
        wandb.config = {"learning_rate": 0.0001, "epochs": args.embedder_epoch, "batch_size": args.embedder_global_batch_size}

    os.makedirs(args.embedder_results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.embedder_results_dir}/*"))
    model_string_name = "vision_encoder"
    experiment_dir = f"{args.embedder_results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    model = WaveletModel( input_channels=4,hidden_dim=args.embedder_embed_dim).to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained("pretrain").to(device)

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    train_dataset = PatientDataGenerator(
        '/mnt/c/Users/CFP',
        '/mnt/c/Users/FFA',
        transform=transforms.Compose([
            transforms.CenterCrop([1388, 1036]),
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    update_ema(ema, model, decay=0)
    ema.eval()
    model.train()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    logger.info(f"Training for {args.embedder_epoch} epochs...")
    for epoch in range(args.embedder_epoch):
        logger.info(f"Beginning epoch {epoch}...")
        item=0
        for cfp, ffa, filename in train_loader:
            item += 1
            cfp = cfp.to(device)
            with torch.no_grad():
                cfp = vae.encode(cfp).latent_dist.sample().mul_(0.18215)
            opt.zero_grad()
            with autocast(enabled=args.autocast):
                x = model(cfp)
            loss = infoNCE_loss_b(x) + 0.6*model.loss_energy + 0.04*model.loss_grad
            if args.wandb:
                wandb.log({"loss": loss.item()})

            if torch.isnan(loss).any():
                logger.info(f"nan...      ignore losses....")
                continue

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.8f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.embedder_ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.", default=False)
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.", default=False)
    parser.add_argument("--use-mamba2", action="store_true", help="if you want use mamba2.", default=False)
    parser.add_argument('--config', type=str, help='Path to the configuration file',
                        default='config/CFPFFA.yaml')
    args = parser.parse_args()
    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)