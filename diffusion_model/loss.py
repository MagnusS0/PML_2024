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

def weight_function(self, t):
    """
    Importance sampling weight function based on E[L_t^2] using a history.
    """
    t = t.squeeze().long()
    weights = []
    for timestep in t:
        history = self.loss_squared_history[timestep.item()]
        if len(history) > 0:
            # Compute E[L_t^2] as the mean of the last 10 values
            weights.append(torch.sqrt(torch.mean(torch.tensor(history))))
        else:
            # Default weight if no history is available
            weights.append(torch.tensor(1.0))
    return torch.tensor(weights, device=t.device)

def update_loss_squared_history(self, t, loss):
    """
    Update the history of L_t^2 for each timestep.
    """
    t = t.squeeze().long()
    # Calculate L_t^2 for the given timesteps
    loss_squared = loss.mean(dim=1) ** 2  # Mean over batch dimension
    for i in range(t.shape[0]):
        timestep = t[i].item()
        if timestep in self.loss_squared_history:
            # Update the history with a new value
            self.loss_squared_history[timestep].append(loss_squared[i].item())
            # Keep only the last 10 values
            if len(self.loss_squared_history[timestep]) > 10:
                self.loss_squared_history[timestep].pop(0)

def elbo_IS(self, x0, weight_function=weight_function):
    """
    ELBO training objective with importance sampling.

    Parameters
    ----------
    x0: torch.tensor
        Input image

    Returns
    -------
    float
        ELBO value
    """

    t = torch.randint(1, self.T, (x0.shape[0], 1)).to(x0.device)
    # Importance weight for each timestep
    weights = weight_function(self, t.float())
    weights = weights / weights.sum()  # Normalize weights

    # Sample noise
    epsilon = torch.randn_like(x0)
    # Forward diffusion to produce image at step t
    xt = self.forward_diffusion(x0, t, epsilon)
    # Compute loss for predicting noise
    loss = nn.MSELoss(reduction='none')(epsilon, self.network(xt, t))
    # Update history of L_t^2
    update_loss_squared_history(self, t, loss)
    # Apply importance weights
    loss = (loss.mean(dim=1) * weights.squeeze()).mean()
    return -loss