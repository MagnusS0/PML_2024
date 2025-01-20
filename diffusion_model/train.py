import torch
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms, utils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from unet import ScoreNet
from ddpm import DDPM
from sde_diffusion import SDEDiffusion
from ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

scaler = GradScaler()

# Create directories for logs and checkpoints
log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
checkpoint_dir = 'checkpoints'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

def train(model, optimizer, scheduler, dataloader, epochs, device, ema=True, per_epoch_callback=None):
    """
    Training loop
    
    Parameters
    ----------
    model: nn.Module
        Pytorch model
    optimizer: optim.Optimizer
        Pytorch optimizer to be used for training
    scheduler: optim.LRScheduler
        Pytorch learning rate scheduler
    dataloader: utils.DataLoader
        Pytorch dataloader
    epochs: int
        Number of epochs to train
    device: torch.device
        Pytorch device specification
    ema: Boolean
        Whether to activate Exponential Model Averaging
    per_epoch_callback: function
        Called at the end of every epoch
    """

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Setup progress bar
    total_steps = len(dataloader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    if ema:
        ema_global_step_counter = 0
        ema_steps = 10
        ema_adjust = dataloader.batch_size * ema_steps / epochs
        ema_decay = 1.0 - 0.995
        ema_alpha = min(1.0, (1.0 - ema_decay) * ema_adjust)
        ema_model = ExponentialMovingAverage(model, device=device, decay=1.0 - ema_alpha)                
    
    for epoch in range(epochs):

        # Switch to train mode
        model.train()

        global_step_counter = 0
        epoch_loss = 0.0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                loss = model.loss(x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

            # Log training metrics
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], global_step)

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter%ema_steps==0:
                    ema_model.update_parameters(model)                
        
        # Log average epoch loss
        writer.add_scalar('Training/Epoch_Loss', epoch_loss / len(dataloader), epoch)

        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model, epoch, writer)

    writer.close()
    return writer, ema_model.module if ema else model

# Parameters
T = 1000
learning_rate = 1e-3
epochs = 100
batch_size = 256
model_type = "DDPM" #SDE or DDPM

# Rather than treating MNIST images as discrete objects, as done in Ho et al 2020, 
# we here treat them as continuous input data, by dequantizing the pixel values (adding noise to the input data)
# Also note that we map the 0..255 pixel values to [-1, 1], and that we process the 28x28 pixel values as a flattened 784 tensor.
if model_type == "DDPM":
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),    # Dequantize pixel values
        transforms.Lambda(lambda x: (x-0.5)*2.0),                    # Map from [0,1] -> [-1, -1]
        transforms.Lambda(lambda x: x.flatten())
    ])
if model_type == "SDE":
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),    # Dequantize pixel values
        transforms.Lambda(lambda x: (x-0.5)*2.0)                     # Map from [0,1] -> [-1, -1]
    ])


# Download and transform train dataset
dataloader_train = torch.utils.data.DataLoader(datasets.MNIST('./data/mnist_data', download=True, train=True, transform=transform),
                                                batch_size=batch_size,
                                                num_workers=4,
                                                pin_memory=True,
                                                shuffle=True,
                                                drop_last=True)

dataloader_test = torch.utils.data.DataLoader(datasets.MNIST('./data/mnist_data', download=True, train=True, transform=transform),
                                                batch_size=1024,
                                                num_workers=4,
                                                pin_memory=True,
                                                shuffle=False)

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_type == "SDE":
    score_net = ScoreNet(lambda t: torch.ones_like(t))  # Will be replaced by SDE's marginal_prob_std
    model = SDEDiffusion(score_net, sampling_method='euler').to(device)

if model_type == "DDPM":
    # Construct Unet
    # The original ScoreNet expects a function with std for all the
    # different noise levels, such that the output can be rescaled.
    # Since we are predicting the noise (rather than the score), we
    # ignore this rescaling and just set std=1 for all t.
    mnist_unet = ScoreNet((lambda t: torch.ones(1).to(device)))

    # Construct model
    model = DDPM(mnist_unet, T=T, beta_schedule="linear", loss_type="IS").to(device)

# Construct optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Setup simple scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)


def reporter(model, epoch, writer):
    """Callback function used for plotting images during training"""
    model.eval()
    with torch.no_grad():
        nsamples = 10
        if model_type == "SDE":
            samples = model.sample((nsamples, 1, 28, 28)).cpu()
        else:
            samples = model.sample((nsamples,28*28)).cpu()
        
        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples+1)/2 
        samples = samples.clamp(0.0, 1.0)

        # Log images to TensorBoard
        if model_type == "SDE":
            grid = utils.make_grid(samples, nrow=nsamples)
        else:
            grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        writer.add_image('Generated_Samples', grid, epoch)
        
        # Plot in grid
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.savefig(f"sample_{epoch}.png")
        plt.close()

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
            if isinstance(model, DDPM):
                real_images = real_images.view(-1, 28 * 28)
            real_images = (real_images + 1) / 2  
            real_images = real_images.repeat(1, 3, 1, 1)  # Repeat channels to get (N, 3, 28, 28)
            real_images = transform(real_images) # Resize to (N, 3, 299, 299)
            fid.update(real_images.to(device), real=True)

    
    fid_score = fid.compute()

    return float(fid_score)

def calculate_is(model, dataloader, device, num_samples=None, num_steps=10):
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
    # Call training loop
    writer, model = train(model, optimizer, scheduler, dataloader_train,
        epochs=epochs, device=device, per_epoch_callback=reporter)
    
    # Calculate and log final metrics
    print("Calculating final metrics...")
    fid_score = calculate_fid(model, dataloader_test, device, num_samples=10000)
    #is_mean, is_std = calculate_is(model, dataloader_train, device, num_samples=1024, num_steps=5)
    
    # Log final metrics to TensorBoard
    writer.add_hparams(
        {'learning_rate': learning_rate, 'epochs': epochs, 'batch_size': batch_size},
        {
            'FID': fid_score,
           # 'IS_mean': is_mean,
            #'IS_std': is_std,
        }
    )
    
    # Save final model
    checkpoint_path = os.path.join(checkpoint_dir, f'model_ema{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'fid_score': fid_score,
    }, checkpoint_path)
    
    print(f"Final FID score: {fid_score:.2f}")
   # print(f"Inception Score: Mean={is_mean}, Std={is_std}")
    print(f"Model saved to {checkpoint_path}")