import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
from loguru import logger
import os
import wandb
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torch.cuda.amp import GradScaler, autocast
from model import DiffMa_models
from block.CFP_encoder import CFP_Encoder
from omegaconf import OmegaConf
from loader_train import PatientDataGenerator
from Wavelet import WaveletModel
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def find_model_model(model_name):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["model"]
    return checkpoint


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logger.add(f"{logging_dir}/log.txt ")
    return logger


def cosine_similarity_loss(original_features, generated_features):
    original_features = original_features / original_features.norm(dim=-1, keepdim=True)
    generated_features = generated_features / generated_features.norm(dim=-1, keepdim=True)
    loss = 1 - F.cosine_similarity(original_features, generated_features).mean()
    return loss


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    scaler = GradScaler()
    torch.manual_seed(args.global_seed)

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    if args.wandb:
        wandb.init(project=args.model.replace('/', '_'))
        wandb.config = {
            "learning_rate": 0.0001,
            "epochs": args.epochs,
            "batch_size": args.global_batch_size,
            "dt-rank": args.dt_rank,
            "d-state": args.d_state,
            "save-path": experiment_dir,
            "autocast": args.autocast,
        }
    logger.info(f"Experiment directory created at {experiment_dir}")

    # 创建模型
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiffMa_models[args.model](
        input_size=latent_size,
        dt_rank=args.dt_rank,
        d_state=args.d_state,
        use_mamba2=args.use_mamba2,
    ).to(device)

    if args.init_from_pretrain_ckpt:
        model_state_dict_ = find_model_model(args.pretrain_ckpt_path)
        model.load_state_dict(model_state_dict_)
        ema = deepcopy(model).to(device)
        ema_state_dict_ = find_model(args.pretrain_ckpt_path)
        ema.load_state_dict(ema_state_dict_)
        logger.info(f"Loaded pretrain model from {args.pretrain_ckpt_path}")
    else:
        ema = deepcopy(model).to(device)

    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained('pretrain').to(device)

    cfpprocessor = WaveletModel(
        input_channels=4,
        hidden_dim=512,
    ).to(device)
    ffapro_ckpt_path = args.cfpprocessor_ckpt
    ffapro_state_dict = find_model(ffapro_ckpt_path)
    cfpprocessor.load_state_dict(ffapro_state_dict)
    cfpprocessor.eval()

    # 加载 CT 编码器
    cfp_encoder = CFP_Encoder(
        img_size=args.image_size // 8,
        patch_size=int(args.model[-1]),
        in_channels=4,
        embed_dim=512,
        contain_mask_token=True,
    ).to(device)
    ct_ckpt_path = args.ct_ckpt
    ct_state_dict = find_model(ct_ckpt_path)
    cfp_encoder.load_state_dict(ct_state_dict)
    cfp_encoder.eval()

    logger.info(f"DiffMa Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Use half-precision training? {args.autocast}")

    # 设置优化器
    lr = args.lr_ if args.init_from_pretrain_ckpt else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    # 创建数据加载器
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

    logger.info(f"Dataset contains {len(train_dataset)}.")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()


    train_steps = args.init_train_steps if args.init_from_pretrain_ckpt else 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")
    epoch_list = []

    save_dir = "feature_visualizations1"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        item = 0
        for cfp, ffa, filename in train_loader:
            item += 1
            cfp = cfp.to(device)
            ffa = ffa.to(device)
            with torch.no_grad():
                if not torch.all((ffa >= -1) & (ffa <= 1)):
                    ffa = ((ffa - ffa.min()) * 1.0 / (ffa.max() - ffa.min())) * 2.0 - 1.0
                ffa = vae.encode(ffa).latent_dist.sample().mul_(0.18215)
                x_ = vae.encode(cfp).latent_dist.sample().mul_(0.18215)
                c1 = cfpprocessor(x_)
                c2 = cfp_encoder(x_)
            t = torch.randint(0, diffusion.num_timesteps, (ffa.shape[0],), device=device)
            model_kwargs = dict(y=c1, y2=c2)

            with autocast(enabled=args.autocast):
                loss_dict = diffusion.training_losses(model, ffa, t,model_kwargs )
                loss = loss_dict["loss"].mean()

            if args.wandb:
                wandb.log({"loss": loss.item()})

            if torch.isnan(loss).any():
                logger.info(f"nan...... ignore losses......")
                continue

            with autocast(enabled=args.autocast):
                scaler.scale(loss).backward()

            if train_steps % args.accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                update_ema(ema, model)
                opt.zero_grad()

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            epoch_list.append(epoch + 1)

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(
                    f"(Epoch={epoch:05d}, step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    logger.info("Done!")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable WandB.", default=False)
    parser.add_argument("--autocast", action="store_true", help="Whether to use half-precision training.",
                        default=False)
    parser.add_argument("--use-mamba2", action="store_true", help="if you want use mamba2.", default=False)
    parser.add_argument('--config', type=str, help='Path to the configuration file',
                        default="config/CFPFFA.yaml")
    args = parser.parse_args()

    cli_config = OmegaConf.create({k: v for k, v in args.__dict__.items() if v is not None and k != 'config'})
    args = OmegaConf.merge(OmegaConf.load(args.config), cli_config)
    main(args)
