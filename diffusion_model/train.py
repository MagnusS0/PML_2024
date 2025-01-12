import torch
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms, utils
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import math
from unet import ScoreNet
from ddpm import DDPM
from ema import ExponentialMovingAverage

scaler = GradScaler()

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
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                loss = model.loss(x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", lr=f"{scheduler.get_last_lr()[0]:.2E}")
            progress_bar.update()

            if ema:
                ema_global_step_counter += 1
                if ema_global_step_counter%ema_steps==0:
                    ema_model.update_parameters(model)                
        
        if per_epoch_callback:
            per_epoch_callback(ema_model.module if ema else model)


# Parameters
T = 1000
learning_rate = 1e-3
epochs = 100
batch_size = 256


# Rather than treating MNIST images as discrete objects, as done in Ho et al 2020, 
# we here treat them as continuous input data, by dequantizing the pixel values (adding noise to the input data)
# Also note that we map the 0..255 pixel values to [-1, 1], and that we process the 28x28 pixel values as a flattened 784 tensor.
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),    # Dequantize pixel values
    transforms.Lambda(lambda x: (x-0.5)*2.0),                    # Map from [0,1] -> [-1, -1]
    transforms.Lambda(lambda x: x.flatten())
])

# Download and transform train dataset
dataloader_train = torch.utils.data.DataLoader(datasets.MNIST('./data/mnist_data', download=True, train=True, transform=transform),
                                                batch_size=batch_size,
                                                num_workers=4,
                                                pin_memory=True,
                                                shuffle=True)

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Construct Unet
# The original ScoreNet expects a function with std for all the
# different noise levels, such that the output can be rescaled.
# Since we are predicting the noise (rather than the score), we
# ignore this rescaling and just set std=1 for all t.
mnist_unet = ScoreNet((lambda t: torch.ones(1).to(device)))

# Construct model
model = DDPM(mnist_unet, T=T).to(device)

# Construct optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Setup simple scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)


def reporter(model):
    """Callback function used for plotting images during training"""
    
    # Switch to eval mode
    model.eval()

    with torch.no_grad():
        nsamples = 10
        samples = model.sample((nsamples,28*28)).cpu()
        
        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples+1)/2 
        samples = samples.clamp(0.0, 1.0)

        # Plot in grid
        grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nsamples)
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.close()

def calculate_fid(model, dataloader, device, num_samples=128):
    """Calculate FID score between real and generated images."""
    # Initialize FID metric
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Define transformation for resizing
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
    ])
    
    # Generate fake images
    model.eval()
    with torch.no_grad():
        fake_samples = []
        for _ in tqdm(range(math.ceil(num_samples / 128)), desc="Generating fake samples"):
            samples = model.sample((128, 28 * 28))
            samples = samples.view(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28)
            samples = (samples + 1) / 2 # Map from [-1, 1] to [0, 1]
            samples = samples.repeat(1, 3, 1, 1)  
            samples = transform(samples)  # Resize to (N, 3, 299, 299)
            fake_samples.append(samples)
        fake_samples = torch.cat(fake_samples)[:num_samples]
        fid.update(fake_samples.to(device), real=False)

    # Process real images
    for real_images, _ in dataloader:
        if fid.real_features_num_samples >= num_samples:
            break
        real_images = real_images.view(-1, 1, 28, 28)  
        real_images = (real_images + 1) / 2  
        real_images = real_images.repeat(1, 3, 1, 1)  # Repeat channels to get (N, 3, 28, 28)
        real_images = transform(real_images) # Resize to (N, 3, 299, 299)
        fid.update(real_images.to(device), real=True)

    # Compute FID
    return float(fid.compute())

if __name__ == "__main__":
    # Call training loop
    train(model, optimizer, scheduler, dataloader_train,
        epochs=epochs, device=device, per_epoch_callback=reporter)
    
    # Calculate FID score after training
    print("Calculating FID score...")
    fid_score = calculate_fid(model, dataloader_train, device)
    print(f"Final FID score: {fid_score:.2f}")