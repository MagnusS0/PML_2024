import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from ddpm import DDPM
from sde_diffusion import SDEDiffusion
from unet import ScoreNet
from likelihood import normal_kl, discretized_gaussian_log_likelihood, ode_likelihood

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
            #if isinstance(model, DDPM):
            real_images = real_images.view(-1, 1, 28, 28)  
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

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def compute_nll_one_batch(model, x0_batch):
    """
    Compute the sum of KL divergences for each diffusion step
    plus the final negative log-likelihood (NLL) term.

    Mathematically, this code implements the variational bound
    on -log p(x_0) in a discrete-time diffusion model:
    
        E[-log p(x_0)] <= Sum_{t=2..T} KL[ q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t) ]
                        + (- log p_theta(x_0|x_1)).

    The returned value is a per-example (in bits per dimension).
    """
    device = x0_batch.device
    batch_size = x0_batch.size(0)
    
    T = model.T
    alpha_bar = model.alpha_bar
    alpha = model.alpha
    beta = model.beta
    
    kl_sum = torch.zeros(batch_size, device=device)
    x_t = [None] * (T + 1)
    x_t[0] = x0_batch

    # Forward process: 
    # Simulate q(x_t | x_{t-1})
    for t in range(1, T + 1):
        epsilon = torch.randn_like(x0_batch)
        t_batch = torch.full((batch_size, 1), t, device=device)
        x_t[t] = model.forward_diffusion(x0_batch, t_batch, epsilon)

    # Compute KL: 
    # KL[ q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t) ] for t = T..1
    for t in range(T, 0, -1):
        t_batch = torch.full((batch_size, 1), t, device=device).float()
        
        # Posterior q(x_{t-1} | x_t, x_0):
        posterior_mean = (
            torch.sqrt(alpha_bar[t-1]) * beta[t] / (1 - alpha_bar[t]) * x0_batch +
            torch.sqrt(alpha[t]) * (1 - alpha_bar[t-1]) / (1 - alpha_bar[t]) * x_t[t]
        )
        posterior_variance = ((1 - alpha_bar[t-1]) / (1 - alpha_bar[t])) * beta[t]
        posterior_logvar = torch.log(torch.clamp(posterior_variance, min=1e-20))

        # p_theta(x_{t-1} | x_t)
        pred_noise = model.network(x_t[t], t_batch)
        pred_mean = (1 / torch.sqrt(alpha[t])) * (
            x_t[t] - ((1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])) * pred_noise
        )
        pred_variance = beta[t]
        pred_logvar = torch.log(torch.clamp(pred_variance, min=1e-20))

        # KL divergence
        kl_t = normal_kl(
            posterior_mean.view(batch_size, -1),
            posterior_logvar.expand(batch_size, 784),
            pred_mean.view(batch_size, -1),
            pred_logvar.expand(batch_size, 784)
        )
        per_example_kl = mean_flat(kl_t) / np.log(2.0)  # Convert to bits
        kl_sum += per_example_kl

    # Final term: - log p_theta(x_0 | x_1)
    x1 = x_t[1]
    t_one = torch.ones((batch_size, 1), device=device)
    pred_noise = model.network(x1, t_one)
    
    final_mean = (1 / torch.sqrt(alpha[1])) * (
        x1 - ((1 - alpha[1]) / torch.sqrt(1 - alpha_bar[1])) * pred_noise
    )
    final_log_scales = 0.5 * torch.log(beta[1].expand(batch_size, 784))
    
    # log p(x_0 | x_1) 
    lls = - discretized_gaussian_log_likelihood(
        x=x0_batch,
        means=final_mean,
        log_scales=final_log_scales
    )
    ll_sum = mean_flat(lls) / np.log(2.0)  # Convert to bits

    return kl_sum + ll_sum

def sample_and_print_images(model, nsamples=12, nrow=6, model_name='test', model_type="SDE"):
    model.eval()
    with torch.no_grad():
        if model_type == "SDE":
            samples = model.sample((nsamples, 1, 28, 28)).cpu()
        else:
            samples = model.sample((nsamples, 28*28)).cpu()
        
        # Map pixel values back from [-1,1] to [0,1]
        samples = (samples + 1) / 2 
        samples = samples.clamp(0.0, 1.0)

        # Create grid
        if model_type == "SDE":
            grid = utils.make_grid(samples, nrow=nrow)
        else:
            grid = utils.make_grid(samples.reshape(-1, 1, 28, 28), nrow=nrow)
        
        # Plot in grid and save
        plt.figure(figsize=(nrow, nsamples // nrow))
        plt.gca().set_axis_off()
        plt.imshow(transforms.functional.to_pil_image(grid), cmap="gray")
        plt.savefig(f"./results/{model_name}_tets_samp.png")
        plt.close()

def load_model(checkpoint_path, loss_type='simple'):
    """Load model with specific configuration"""
    model = DDPM(
        network=ScoreNet(lambda t: torch.ones_like(t)), 
        T=1000,
        loss_type=loss_type
    )

    if loss_type == 'SDE':
        model = SDEDiffusion(
            network=ScoreNet(lambda t: torch.ones_like(t)),
            sampling_method='euler'
        )
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    return model

if __name__ == "__main__":
    configs = [
        # ("./checkpoints/elbo_new_cosine_simple.pt", "simple"),
        # ("./checkpoints/elbo_new_cosine_lds.pt", "LDS"),
        # ("./checkpoints/elbo_new_cosine_is.pt", "IS"),
        # ("./checkpoints/elbo_simple_linear.pt", "simple"),
        # ("./checkpoints/elbo_LDS_linear.pt", "LDS"),
        # ("./checkpoints/elbo_is_linear.pt", "IS"),
        # ("./checkpoints/elbo_is_cosine.pt", "IS"),
        # ("./checkpoints/elbo_lds_cosine.pt", "LDS"),
        # ("./checkpoints/elbo_simple_cosine.pt", "simple"),
        # ("./checkpoints/model_ema20250116_211447.pt", "LDS"),
        # ("./checkpoints/model_ema20250116_233113.pt", "simple"),
        ("checkpoints/model_ema20250116_222311.pt", "SDE"),
        #("./checkpoints/model_ema20250117_000807.pt", "IS"),
    ]

    # Load data
    transform = get_transform("SDE")

    dataloader_train = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist_data', download=True, train=True, transform=transform),
        batch_size=1024,
        num_workers=4,
        pin_memory=False,
        shuffle=True
)
    
    for checkpoint_path, loss_type in configs:
        print(f"\nEvaluating model with {loss_type} loss:")
        model = load_model(checkpoint_path, loss_type)
        
    #     total_nll = 0.
    #     total_count = 0

    #     model.eval()
    #     with torch.no_grad():
    #         for x0_batch, _ in tqdm(dataloader_train):
    #             x0_batch = x0_batch.to('cuda')
    #             nll = compute_nll_one_batch(model, x0_batch)
    #             total_nll += nll.sum().item()
    #             total_count += nll.size(0)

    #     avg_nll = total_nll / total_count
    #     print(f"Average NLL: {avg_nll}")

        all_bpds = 0.
        all_items = 0
        tqdm_data = tqdm(dataloader_train)
        for x, _ in tqdm_data:
            x = x.to('cuda')
            # uniform dequantization
            x = (x * 255. + torch.rand_like(x)) / 256.    
            _, bpd = ode_likelihood(x, model, model.marginal_prob_std,
                                    model.diffusion_coeff,
                                    x.shape[0], device=('cuda'), eps=1e-5)
            all_bpds += bpd.sum()
            all_items += bpd.shape[0]
            tqdm_data.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))

        # # Calculate FID and IS
        # fid = calculate_fid(model, dataloader_train, 'cuda')
        # # # is_mean, is_std = calculate_is(model, dataloader_train, 'cuda')

        # print(f"FID: {fid}")

        # Generate images
        sample_and_print_images(model, model_name=f"model_{loss_type}", model_type="SDE", nsamples=12)
        print(f"Images generated for {loss_type} loss")