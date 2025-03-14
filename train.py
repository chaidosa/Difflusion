#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Train SEINE diffusion model on custom dataset.
"""
import os
import sys
import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
from PIL import Image
import glob
from pathlib import Path

try:
    import utils
    from diffusion import create_diffusion
except:
    sys.path.append(os.path.split(sys.path[0])[0])
    import utils
    from diffusion import create_diffusion

from models import get_models, get_lr_scheduler
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from datasets import video_transforms

class VideoDataset(Dataset):
    def __init__(self, data_root, frame_count=24, image_size=(256, 256), default_caption="collsion of rigid objects"):
        self.data_root = data_root
        self.frame_count = frame_count
        # Ensure image_size is a standard tuple (not OmegaConf)
        self.image_size = tuple(image_size) if hasattr(image_size, '__iter__') else (image_size, image_size)
        self.default_caption = default_caption
        
        # Check if data path exists
        data_path = Path(data_root)
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_root}")
        
        print(f"Searching for scenes in: {data_root}")
        
        # List all items in the directory for debugging
        all_items = list(data_path.iterdir())
        print(f"Total items in directory: {len(all_items)}")
        for item in all_items:
            print(f"  Found item: {item.name} (is_dir: {item.is_dir()})")
            
        # Find all scene directories
        self.scene_paths = sorted([d for d in all_items if d.is_dir()])
        print(f"Found {len(self.scene_paths)} scenes in {data_root}")
        
        # Video transforms
        # Ensure we pass regular tuple to ResizeVideo
        self.transform = transforms.Compose([
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.ResizeVideo(self.image_size),  # Using self.image_size which is guaranteed to be a tuple
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    def __len__(self):
        return len(self.scene_paths)
    
    def __getitem__(self, idx):
        scene_path = self.scene_paths[idx]
        scene_id = scene_path.name
        
        # Get frame files in the scene directory
        frame_files = sorted(list(scene_path.glob("*.jpg")) + list(scene_path.glob("*.png")))
        
        # Ensure we have at least required number of frames
        if len(frame_files) < self.frame_count:
            raise ValueError(f"Scene {scene_id} has only {len(frame_files)} frames, but {self.frame_count} are required")
        
        # Select a random subset if we have more frames than needed or use all if exact match
        if len(frame_files) > self.frame_count:
            # Option 1: Random consecutive sequence
            start_idx = np.random.randint(0, len(frame_files) - self.frame_count + 1)
            frame_files = frame_files[start_idx:start_idx + self.frame_count]
            
            # Option 2: Evenly spaced frames (uncomment to use)
            # indices = np.linspace(0, len(frame_files) - 1, self.frame_count, dtype=int)
            # frame_files = [frame_files[i] for i in indices]
        
        # Load frames
        video_frames = []
        for frame_file in frame_files:
            img = Image.open(frame_file).convert("RGB")
            img_tensor = torch.as_tensor(np.array(img, dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(img_tensor)
        
        # Stack and transform
        video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)  # f,c,h,w
        video_frames = self.transform(video_frames)
        
        # Use scene_id for caption
        caption = f"{self.default_caption} {scene_id}"
        
        return {
            "video": video_frames,
            "text": caption,
            "scene_id": scene_id
        }

def collate_fn(batch):
    videos = torch.stack([item["video"] for item in batch])
    captions = [item["text"] for item in batch]
    scene_ids = [item["scene_id"] for item in batch]
    
    return {
        "video": videos,
        "text": captions,
        "scene_id": scene_ids
    }

def train(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(True)
    device = torch.device(args.device)
    
    # Convert OmegaConf objects to standard Python types where needed
    # This prevents type errors with torch functions
    if hasattr(args, 'image_size'):
        args.image_size = tuple(args.image_size)
    
    # Setup data loader
    # Convert OmegaConf ListConfig to regular tuple for image_size
    image_size = tuple(args.image_size) if hasattr(args, 'image_size') else (256, 256)
    
    dataset = VideoDataset(
        data_root=args.data_path,
        frame_count=args.num_frames,
        image_size=image_size,
        default_caption=args.default_caption if hasattr(args, 'default_caption') else "video scene"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Initialize model:
    # Convert image_size to Python types to avoid OmegaConf issues
    image_h = int(args.image_size[0])
    image_w = int(args.image_size[1])
    latent_h = image_h // 8
    latent_w = image_w // 8
    args.latent_h = latent_h
    args.latent_w = latent_w
    args.use_mask = False  # During training, we don't use mask
    
    print('Initializing model...')
    model = get_models(args).to(device)
    
    # Enable xformers memory efficient attention if available
    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
            print("Successfully enabled xformers memory efficient attention")
        else:
            print("WARNING: xformers is not available but was requested. Install it with: pip install xformers")
            
    # Set PyTorch to allocate memory more efficiently
    if torch.cuda.is_available():
        # Enable memory efficient features when possible
        torch.cuda.empty_cache()
        # Try setting memory allocation options
        try:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid memory fragmentation")
        except:
            pass
    
    # Load VAE and Text Encoder from pretrained model
    print('Loading VAE and Text Encoder...')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    text_encoder = TextEmbedder(args.pretrained_model_path).to(device)
    
    # Freeze VAE and Text Encoder
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    vae.eval()
    text_encoder.eval()
    
    # Initialize diffusion
    diffusion = create_diffusion(
        timestep_respacing=str(args.diffusion_steps),
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2)
    )
    
    # Setup learning rate scheduler
    scheduler_params = {}
    if args.lr_scheduler == 'warmup':
        scheduler_params['warmup_steps'] = args.warmup_steps
    elif args.lr_scheduler == 'cosine':
        scheduler_params['T_max'] = args.total_steps
    
    lr_scheduler = get_lr_scheduler(
        optimizer,
        name=args.lr_scheduler,
        **scheduler_params
    )
    
    # Setup EMA
    ema_model = None
    if args.ema_rate > 0:
        ema_model = get_models(args).to(device)
        ema_model.load_state_dict(model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
    
    # Optionally resume from checkpoint
    start_step = 0
    if args.resume_checkpoint:
        print(f'Resuming from checkpoint: {args.resume_checkpoint}')
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'ema' in checkpoint and ema_model is not None:
            ema_model.load_state_dict(checkpoint['ema'])
        start_step = checkpoint['step'] + 1
        
    # Training loop
    print('Starting training...')
    model.train()
    step = start_step
    
    best_loss = float('inf')
    
    while step < args.total_steps:
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Step {step}/{args.total_steps}")):
            if step >= args.total_steps:
                break
                
            # Move data to device
            video = batch["video"].to(device)
            text_prompts = batch["text"]
            
            b, f, c, h, w = video.shape
            
            # Encode video frames to latent space
            with torch.no_grad():
                video_flat = rearrange(video, 'b f c h w -> (b f) c h w')
                latent = vae.encode(video_flat).latent_dist.sample().mul_(0.18215)
                latent = rearrange(latent, '(b f) c h w -> b c f h w', b=b)
                
                # Encode text prompts
                text_embeds = text_encoder(text_prompts=text_prompts, train=False)
                
                # Fix dimensionality: Add an extra dimension to match model expectations
                # Change from [batch, seq_len, hidden_size] to [batch, 1, seq_len, hidden_size]
                # This matches what the model expects without changing its code
                text_embeds = text_embeds.unsqueeze(1)
            
            # Sample noise
            noise = torch.randn_like(latent)
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (b,), device=device
            ).long()
            
            # Add noise to latents according to noise schedule
            noisy_latent = diffusion.q_sample(latent, timesteps, noise=noise)
            
            # Predict noise
            model_kwargs = dict(
                encoder_hidden_states=text_embeds,  # text_embeds already has the extra dimension we added
                class_labels=None
            )
            
            # Forward pass
            model_output = model(noisy_latent, timesteps, **model_kwargs)
            
            # Calculate loss
            if args.predict_xstart:
                target = latent
            else:
                target = noise
            
            # Extract the sample tensor from UNet3DConditionOutput object
            # The model returns an object with a 'sample' attribute containing the actual tensor
            if hasattr(model_output, 'sample'):
                model_output_tensor = model_output.sample
            else:
                model_output_tensor = model_output
                
            loss = F.mse_loss(model_output_tensor, target, reduction="mean")
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
            optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Update EMA model
            if ema_model is not None:
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(args.ema_rate).add_(p_model.data, alpha=1 - args.ema_rate)
            
            # Log progress
            epoch_loss += loss.item()
            num_batches += 1
            
            if step % args.log_interval == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.7f}")
            
            # Save checkpoint
            if step > 0 and step % args.save_interval == 0:
                checkpoint = {
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                }
                
                if lr_scheduler is not None:
                    checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
                if ema_model is not None:
                    checkpoint['ema'] = ema_model.state_dict()
                
                # Regular checkpoint
                torch.save(
                    checkpoint,
                    os.path.join(args.output_dir, "checkpoints", f"model_{step:08d}.pt")
                )
                
                # Save best model
                avg_loss = epoch_loss / (num_batches if num_batches > 0 else 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(
                        checkpoint,
                        os.path.join(args.output_dir, "checkpoints", "model_best.pt")
                    )
                
                print(f"Saved checkpoint at step {step}")
            
            step += 1
    
    # Save final model
    final_checkpoint = {
        'step': step - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
    }
    
    if lr_scheduler is not None:
        final_checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    if ema_model is not None:
        final_checkpoint['ema'] = ema_model.state_dict()
    
    torch.save(
        final_checkpoint,
        os.path.join(args.output_dir, "checkpoints", "model_final.pt")
    )
    
    print(f"Training completed! Final model saved at {os.path.join(args.output_dir, 'checkpoints', 'model_final.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    parser.add_argument("--data_path", type=str, help="Override data path from config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments if provided
    if args.data_path:
        print(f"Overriding data_path from command line: {args.data_path}")
        config.data_path = args.data_path
        
    train(config)