import torch
import torch.nn as nn

def elbo_simple(model, x0):
    """
    ELBO training objective (Algorithm 1 in Ho et al, 2020)

    Parameters
    ----------
    model: DDPM
        The DDPM model instance
    x0: torch.tensor
        Input image

    Returns
    -------
    float
        ELBO value
    """
    # Sample time step t
    t = torch.randint(0, model.T, (x0.shape[0],1)).to(x0.device)

    # Sample noise
    epsilon = torch.randn_like(x0)

    # Forward diffusion to produce image at step t
    xt = model.forward_diffusion(x0, t, epsilon)

    return -nn.MSELoss(reduction='mean')(epsilon, model.network(xt, t))

def elbo_LDS(model, x0):
    """
    ELBO training objective with low-discrepancy sampler for discrete timesteps.
    """
    batch_size = x0.shape[0]
    u0 = torch.rand(1, device=x0.device)
    
    t_continuous = (u0 + torch.arange(batch_size, device=x0.device) / batch_size) % 1.0
    t_frac = t_continuous * model.T
    t_int = torch.floor(t_frac)
    t_prob_up = t_frac - t_int
    t_rand = torch.rand_like(t_prob_up)
    t_discrete = (t_int + (t_rand < t_prob_up).long()).long()
    t_discrete = t_discrete.unsqueeze(-1)

    epsilon = torch.randn_like(x0)
    xt = model.forward_diffusion(x0, t_discrete, epsilon)

    return -nn.MSELoss(reduction='mean')(epsilon, model.network(xt, t_discrete))

def elbo_LDS_2(model, x0):
    """
    ELBO training objective with low-discrepancy sampler.
    """
    u0 = torch.rand((1,), device=x0.device)
    batch_size = x0.shape[0]
    t = ((u0 + torch.arange(batch_size, device=x0.device) / batch_size) % 1) * model.T
    t = t.long().unsqueeze(1)

    epsilon = torch.randn_like(x0)
    xt = model.forward_diffusion(x0, t, epsilon)

    return -nn.MSELoss(reduction='mean')(epsilon, model.network(xt, t))

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

def elbo_IS(model, x0):
    """
    ELBO training objective with importance sampling.
    """
    # t = torch.randint(1, model.T, (x0.shape[0], 1)).to(x0.device)
    # # Importance weight for each timestep
    # weights = weight_function(model, t.float())
    # weights = weights / weights.sum()  # Normalize weights

    # Sample noise

    weights = torch.tensor([
        torch.sqrt(torch.mean(torch.tensor(model.loss_squared_history[t]))) 
        if len(model.loss_squared_history[t]) > 0 
        else 1.0 
        for t in range(1, model.T)
    ]).to(x0.device)
    
    weights = weights / weights.sum()
    t = torch.multinomial(weights, x0.shape[0], replacement=True).unsqueeze(-1) + 1
    
    epsilon = torch.randn_like(x0)
    xt = model.forward_diffusion(x0, t, epsilon)
    
    loss = nn.MSELoss(reduction='none')(epsilon, model.network(xt, t))
    model.update_loss_squared_history(t, loss)

    loss = (loss.mean(dim=1) * weights[t.squeeze() - 1]).mean()
    
    return -loss.mean()