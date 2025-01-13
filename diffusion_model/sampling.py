import torch
from tqdm import tqdm

def Euler_Maruyama_sampler(score_model, 
                          marginal_prob_std,
                          diffusion_coeff, 
                          batch_size=64, 
                          num_steps=1000, 
                          device='cuda', 
                          eps=1e-3):
    """Generate samples using Euler-Maruyama solver"""
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
        * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    
    with torch.no_grad():
        for time_step in tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
    
    return mean_x