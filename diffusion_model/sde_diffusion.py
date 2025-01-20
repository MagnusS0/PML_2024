# This i built based on the tutorial by Yang Song et al.
# https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import torch
import torch.nn as nn
import numpy as np
from sampling import Euler_Maruyama_sampler, ode_sampler

class SDEDiffusion(nn.Module):
    def __init__(self, network, sigma=25.0, sampling_method='euler'):
        """
        Initialize SDE-based Diffusion Model
        
        Args:
            network: The score model (typically UNet)
            sigma: The sigma parameter for the SDE
        """
        super(SDEDiffusion, self).__init__()
        self.sigma = sigma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Remove the reshaping wrapper and use direct network
        self.network = network
        self.network.marginal_prob_std = self.marginal_prob_std

        # Sampling method
        self.sampling_method = sampling_method
    
    def marginal_prob_std(self, t):
        """Compute standard deviation of p_{0t}(x(t) | x(0))"""
        t = torch.as_tensor(t, device=self.device)
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))
    
    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient"""
        return torch.as_tensor(self.sigma**t, device=self.device)
    
    def forward(self, x, t):
        """Forward pass of the model"""
        return self.network(x, t)
    
    @torch.no_grad()
    def sample(self, shape):
        """
        Generate samples using Euler-Maruyama solver
        
        Args:
            shape: tuple specifying output shape (nsamples, 1, 28, 28)
        """
        if self.sampling_method == 'euler':
            samples = Euler_Maruyama_sampler(
                self,
                self.marginal_prob_std,
                self.diffusion_coeff,
                batch_size=shape[0],
                device=self.device
            )
        elif self.sampling_method == 'ode':
            samples = ode_sampler(
                self,
                self.marginal_prob_std,
                self.diffusion_coeff,
                batch_size=shape[0],
                device=self.device
            )
        return samples
    
    def loss(self, x):
        """Loss function for training score-based models.
        
        Args:
            x: A mini-batch of training data.    
        """
        return self.loss_fn(x)  # Use the SDE-specific loss
    
    def loss_fn(self, x, eps=1e-5):
        """SDE-specific loss function"""
        random_t = torch.rand(x.shape[0], device=self.device) * (1. - eps) + eps  
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)
        
        # Input is already in correct shape (B, 1, 28, 28)
        perturbed_x = x + z * std[:, None, None, None]
        score = self.forward(perturbed_x, random_t)
        
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=[1,2,3]))
        return loss