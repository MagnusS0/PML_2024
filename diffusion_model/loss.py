import torch
import torch.nn as nn

def elbo_simple(self, x0):
    """
    ELBO training objective (Algorithm 1 in Ho et al, 2020)

    Parameters
    ----------
    x0: torch.tensor
        Input image

    Returns
    -------
    float
        ELBO value
    """

    # Sample time step t
    t = torch.randint(0, self.T, (x0.shape[0],1)).to(x0.device)

    # Sample noise
    epsilon = torch.randn_like(x0)

    # TODO: Forward diffusion to produce image at step t
    xt = self.forward_diffusion(x0, t, epsilon)

    return -nn.MSELoss(reduction='mean')(epsilon, self.network(xt, t))

def elbo_LDS(self, x0):
    """
    ELBO training objective with low-discrepancy sampler for discrete timesteps.
    (adapted from Eq. 14 in "Variational Diffusion Models" by Kingma et al., 2021).

    Parameters
    ----------
    x0: torch.tensor
        Input image (batch_size, num_channels, height, width)

    Returns
    -------
    float
        ELBO value
    """

    batch_size = x0.shape[0]

    # Sample random number u0 from U[0,1]
    u0 = torch.rand(1, device=x0.device)

    # Generate low-discrepancy sequence in [0, 1]
    t_continuous = (u0 + torch.arange(batch_size, device=x0.device) / batch_size) % 1.0

    # Map to discrete timesteps [0, T-1] and round
    # Stochastic Rounding
    t_frac = t_continuous * self.T  # Scale to [0, T-1]
    t_int = torch.floor(t_frac)
    t_prob_up = t_frac - t_int
    t_rand = torch.rand_like(t_prob_up)
    t_discrete = (t_int + (t_rand < t_prob_up).long()).long()

    t_discrete = t_discrete.unsqueeze(-1)  # (batch_size, 1)

    # Sample noise
    epsilon = torch.randn_like(x0)

    # Forward diffusion
    xt = self.forward_diffusion(x0, t_discrete, epsilon)

    # Compute loss
    return -nn.MSELoss(reduction='mean')(epsilon, self.network(xt, t_discrete))

def elbo_LDS_2(self, x0): #CAMBIADO
      """
      ELBO training objective with low-discrepancy sampler.

      Parameters
      ----------
      x0: torch.tensor
          Input image

      Returns
      -------
      float
          ELBO value
      """

      # Sample one random number u0 from U[0,1]
      u0 = torch.rand((1,), device=x0.device)

      # Generate k timesteps (for batch size k) using low-discrepancy sampling
      batch_size = x0.shape[0]
      t = ((u0 + torch.arange(batch_size, device=x0.device) / batch_size) % 1) * self.T
      t = t.long().unsqueeze(1)  # Ensure t is integer and of shape (batch_size, 1)

      # Sample noise
      epsilon = torch.randn_like(x0)

      # Forward diffusion to produce image at step t
      xt = self.forward_diffusion(x0, t, epsilon)

      # Compute loss for predicting noise
      return -nn.MSELoss(reduction='mean')(epsilon, self.network(xt, t))