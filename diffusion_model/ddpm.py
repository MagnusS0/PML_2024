import torch
import torch.nn as nn
import math
from tqdm import tqdm
from loss import elbo_simple, elbo_LDS, elbo_LDS_2, elbo_IS

class DDPM(nn.Module):

    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2, beta_schedule='cosine', loss_type='simple'):
        """
        Initialize Denoising Diffusion Probabilistic Model

        Parameters
        ----------
        network: nn.Module
            The inner neural network used by the diffusion process. Typically a Unet.
        beta_1: float
            beta_t value at t=1
        beta_T: [float]
            beta_t value at t=T (last step)
        T: int
            The number of diffusion steps.
        beta_schedule: str
            The schedule for beta. Options: 'cosine', 'linear'
        loss_type: str
            The type of loss to use. Options: 'simple', 'constrained'
        """

        super(DDPM, self).__init__()

        self.loss_type = loss_type

        # Normalize time input before evaluating neural network
        self._network = network
        self.network = lambda x, t: (self._network(x.reshape(-1, 1, 28, 28), 
                                                   (t.squeeze()/T))
                                    ).reshape(-1, 28*28)

        # Total number of time steps
        self.T = T

        # Registering as buffers to ensure they get transferred to the GPU automatically
        if beta_schedule == 'cosine':
            self.register_buffer("beta", self.cosine_variance_schedule(T+1))
        elif beta_schedule == 'linear':
            self.register_buffer("beta", torch.linspace(beta_1, beta_T, T+1))
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        self.register_buffer("alpha", 1-self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

        self.loss_history_size = 10
        self.loss_history = []

    @staticmethod
    def cosine_variance_schedule(timesteps, s=0.008):
        """Cosine schedule from Improved DDPM paper"""
        steps = torch.linspace(0, timesteps, steps=timesteps+1, dtype=torch.float32)
        f_t = torch.cos(((steps/timesteps + s)/(1.0 + s)) * math.pi/2.)**2
        alphas = f_t[1:]/f_t[:timesteps]
        betas = torch.clip(1.0 - alphas, 0.0, 0.999)
        return betas

    def forward_diffusion(self, x0, t, epsilon):
        '''
        q(x_t | x_0)
        Forward diffusion from an input datapoint x0 to an xt at timestep t, provided a N(0,1) noise sample epsilon.
        Note that we can do this operation in a single step

        Parameters
        ----------
        x0: torch.tensor
            x value at t=0 (an input image)
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t
        '''

        # TODO: Define the mean and std variables
        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1 - self.alpha_bar[t])

        return mean + std*epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        """
        p(x_{t-1} | x_t)
        Single step in the reverse direction, from x_t (at timestep t) to x_{t-1}, provided a N(0,1) noise sample epsilon.

        Parameters
        ----------
        xt: torch.tensor
            x value at step t
        t: int
            step index
        epsilon:
            noise sample

        Returns
        -------
        torch.tensor
            image at timestep t-1
        """

        # TODO: Define the mean and std variables
        mean = ( 1 / torch.sqrt(self.alpha[t]) )  *  ( xt - ((1-self.alpha[t])/(torch.sqrt(1-self.alpha_bar[t])))*self.network(xt, t))
        std = ( (1 - self.alpha_bar[t-1]) / (1 - self.alpha_bar[t]) ) * self.beta[t]

        return mean + std*epsilon
    
    def reverse_diffusion_constrained(self, xt, t, epsilon):
        """
        Constrained reverse diffusion that clamps x0 predictions to [-1,1]
        """
        # Predict x0 using xt and the model's predicted noise
        predicted_noise = self.network(xt, t)
        x0 = (xt - torch.sqrt(1 - self.alpha_bar[t]) * predicted_noise) / torch.sqrt(self.alpha_bar[t])
        
        # Clip x0 to [-1, 1]
        x0 = torch.clamp(x0, -1, 1)

        # Now sample xt-1 using the equation for q(xt-1 | xt, x0)
        if t.min() > 1:
            mean = (
                torch.sqrt(self.alpha_bar[t - 1]) * self.beta[t] / (1 - self.alpha_bar[t]) * x0 +
                torch.sqrt(self.alpha[t]) * (1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t]) * xt
            )
            std = torch.sqrt((1 - self.alpha_bar[t-1]) / (1 - self.alpha_bar[t]) * self.beta[t])
            xt_1 = mean + std * epsilon
        else:
            #Special case for t=1
            mean = (1 / torch.sqrt(self.alpha[t])) * (xt - ((1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])) * predicted_noise)
            xt_1 = mean
        
        
        return xt_1


    @torch.no_grad()
    def sample(self, shape):
        """
        Sample from diffusion model (Algorithm 2 in Ho et al, 2020)

        Parameters
        ----------
        shape: tuple
            Specify shape of sampled output. For MNIST: (nsamples, 28*28)

        Returns
        -------
        torch.tensor
            sampled image
        """

        # Sample xT: Gaussian noise
        xT = torch.randn(shape).to(self.beta.device)

        xt = xT
        for t in tqdm(range(self.T, 0, -1)):
            noise = torch.randn_like(xT) if t > 1 else 0
            t = torch.tensor(t).expand(xt.shape[0], 1).to(self.beta.device)
            xt = self.reverse_diffusion_constrained(xt, t, noise)

        return xt
    
    def loss(self, x0):
        """
        Loss function. Just the negative of the ELBO.
        """
        if self.loss_type == 'simple':
            return -elbo_simple(self, x0).mean()
        if self.loss_type == 'LDS':
            return -elbo_LDS(self, x0).mean()
        if self.loss_type == 'LDS_2':
            return -elbo_LDS_2(self, x0).mean()
        if self.loss_type == 'IS':
            return -elbo_IS(self, x0).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def update_loss_squared_history(self, t, loss):
        """
        Update the history of L_t^2 for each timestep.
        """
        t = t.squeeze().long()
        loss_squared = loss.mean(dim=1) ** 2
        for i, t_i in enumerate(t):
            self.loss_squared_history[t_i.item()].append(loss_squared[i].item())


if __name__ == '__main__':
    # Test the DDPM class
    from unet import ScoreNet
    mnist_unet = ScoreNet((lambda t: torch.ones(1).to('cpu')))
    T = 1000
    # Construct model
    model = DDPM(mnist_unet, T=T, beta_schedule="linear", loss_type="IS").to('cpu')
    print(model)
