import torch
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms, utils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.models import inception_v3
from torch.nn.functional import softmax
import numpy as np
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math
from unet import ScoreNet
from ddpm import DDPM
from sde_diffusion import SDEDiffusion
from ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse
from config import Config, default_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--T', type=int, default=default_config['T'])
    parser.add_argument('--learning_rate', type=float, default=default_config['learning_rate'])
    parser.add_argument('--epochs', type=int, default=default_config['epochs'])
    parser.add_argument('--batch_size', type=int, default=default_config['batch_size'])
    parser.add_argument('--model_type', type=str, default=default_config['model_type'])
    parser.add_argument('--beta_schedule', type=str, default=default_config['beta_schedule'])
    parser.add_argument('--loss_type', type=str, default=default_config['loss_type'])
    return parser.parse_args()

def get_transform(model_type):
    """Get the appropriate transform based on model type"""
    if model_type == "DDPM":
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),    # Dequantize pixel values
            transforms.Lambda(lambda x: (x-0.5)*2.0),                    # Map from [0,1] -> [-1, -1]
            transforms.Lambda(lambda x: x.flatten())
        ])
    else:  # SDE
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),    # Dequantize pixel values
            transforms.Lambda(lambda x: (x-0.5)*2.0)                     # Map from [0,1] -> [-1, -1]
        ])

def init_model(config, device):
    """Initialize the appropriate model based on configuration"""
    if config.model_type == "SDE":
        score_net = ScoreNet(lambda t: torch.ones_like(t)).to(device)  # Will be replaced by SDE's marginal_prob_std
        model = SDEDiffusion(score_net).to(device)
    else:  # DDPM
        score_net = ScoreNet(lambda t: torch.ones(1).to(device)).to(device)
        model = DDPM(score_net, T=config.T, beta_schedule=config.beta_schedule, 
                    loss_type=config.loss_type).to(device)
    return model

def reporter(model, epoch, writer, config):
    """Callback function used for plotting images during training"""
    model.eval()
    with torch.no_grad():
        nsamples = 10
        if config.model_type == "SDE":
            samples = model.sample((nsamples, 1, 28, 28)).cpu()
        else:
            samples = model.sample((nsamples,28*28)).cpu()
            
        # Map samples from [-1,1] back to [0,1]
        samples = (samples + 1) * 0.5
        samples = samples.clamp(0.0, 1.0)

        # Log images to TensorBoard
        if config.model_type == "SDE":
            grid = utils.make_grid(samples, nrow=nsamples)
        else:
            grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        writer.add_image('samples', grid, global_step=epoch)
    model.train()

def train(model, optimizer, scheduler, dataloader, device, config,
          ema=True, per_epoch_callback=None):
    """Training loop"""
    writer = SummaryWriter(log_dir)

    total_steps = len(dataloader)*config.epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / config.epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(model, device=device, decay=1.0 - ema_alpha)     

               
    
    for epoch in tqdm(range(config.epochs)):

        model.train()

        global_step_counter = 0
        epoch_loss = 0.0

        for i, (x, _) in enumerate(dataloader):

            x = x.to(device)

            optimizer.zero_grad()
            
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model.loss(x)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter%ema_steps==0:
                    ema_model.update_parameters(model)

            # Log training metrics
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], global_step)

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{config.epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()

            epoch_loss += loss.item()
            global_step_counter += 1
               
        scheduler.step()
        writer.add_scalar('Training/Epoch_Loss', epoch_loss / len(dataloader), epoch)
    
        
        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model, epoch, writer, config)
    
    return writer

def calculate_fid(model, dataloader, device, num_samples=None):
    """Calculate FID score between real and generated images"""
    fid = FrechetInceptionDistance(normalize=True).to(device)

    if not num_samples:
        num_samples = 10000
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
    ])
    
    # Generate fake images
    model.eval()
    with torch.no_grad():
        for real_images, _ in dataloader:
            batch_size = real_images.shape[0]
            if isinstance(model, SDEDiffusion):
                samples = model.sample((batch_size, 28 * 28))
            else:
                samples = model.sample((batch_size, 28 * 28))
                samples = samples.view(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)
            samples = (samples + 1) / 2 # Map from [-1, 1] to [0, 1]
            samples = samples.repeat(1, 3, 1, 1)  
            samples = transform(samples)  # Resize to (N, 3, 299, 299)
            fid.update(samples.to(device), real=False)
            if fid.real_features_num_samples >= num_samples:
                break
            # Get real images
            real_images = real_images.view(-1, 1, 28, 28)  
            real_images = (real_images + 1) / 2  
            real_images = real_images.repeat(1, 3, 1, 1)  # Repeat channels to get (N, 3, 28, 28)
            real_images = transform(real_images) # Resize to (N, 3, 299, 299)
            fid.update(real_images.to(device), real=True)

    
    fid_score = fid.compute()

    return float(fid_score)

def calculate_is(model, dataloader, device, num_samples=None, num_steps=5):
    is_score = InceptionScore(normalize=True).to(device)

    if not num_samples:
        num_samples = 1024

    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
    ])

    # Generate fake images
    model.eval()
    with torch.no_grad():
        for real_images, _ in dataloader:
            batch_size = real_images.shape[0]
            if isinstance(model, SDEDiffusion):
                samples = model.sample((batch_size, 28 * 28))
            else:
                samples = model.sample((batch_size, 28 * 28))
                samples = samples.view(-1, 1, 28, 28)
            samples = (samples + 1) / 2
            samples = samples.repeat(1, 3, 1, 1)
            samples = transform(samples)
            is_score.update(samples.to(device))
        
    is_score = is_score.compute()
    is_mean = is_score[0]
    is_std = is_score[1]

    return is_mean, is_std

if __name__ == "__main__":
    # Parse arguments and create config
    args = parse_args()
    config = Config.from_dict(vars(args))
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get appropriate transform and create dataloader
    transform = get_transform(config.model_type)
    dataloader_train = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist_data', download=True, train=True, transform=transform),
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )
    
    # Initialize model, optimizer, and scheduler
    model = init_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
    
    # Create directories for logs and checkpoints
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize scaler for mixed precision training
    scaler = GradScaler()
    
    # Train model
    writer = train(model, optimizer, scheduler, dataloader_train,
                  device, config, per_epoch_callback=reporter)
    
    # Calculate and log final metrics
    print("Calculating final metrics...")
    fid_score = calculate_fid(model, dataloader_train, device, config.eval_fid_samples)
    is_mean, is_std = calculate_is(model, dataloader_train, device, config.eval_is_samples, config.eval_is_splits)
    
    # Log final metrics to TensorBoard
    writer.add_hparams(
        {'learning_rate': config.learning_rate, 'epochs': config.epochs, 'batch_size': config.batch_size},
        {
            'FID': fid_score,
            'IS_mean': is_mean,
            'IS_std': is_std,
        }
    )
    
    # Save final model
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.epochs,
        'fid_score': fid_score,
        'mean_is': is_mean,
        'std_is': is_std,
        'model_type': config.model_type,
        'beta_schedule': config.beta_schedule,
        'loss_type': config.loss_type
    }, checkpoint_path)
    
    print(f"Final FID score: {fid_score:.2f}")
    print(f"Inception Score: Mean={is_mean}, Std={is_std}")
    print(f"Model saved to {checkpoint_path}")