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

def weight_function(model, t):
    """
    Importance sampling weight function based on E[L_t^2] using a history.
    """
    t = t.squeeze().long()
    weights = []
    for timestep in t:
        history = model.loss_squared_history[timestep.item()]
        if len(history) > 0:
            # Compute E[L_t^2] as the mean of the last 10 values
            weights.append(torch.sqrt(torch.mean(torch.tensor(history))))
        else:
            # Default weight if no history is available
            weights.append(torch.tensor(1.0))
    return torch.tensor(weights, device=t.device)


def elbo_IS(model, x0):
    """
    ELBO training objective with importance sampling of time steps and moving average of loss.

    Parameters
    ----------
    x0: torch.tensor
        Input image

    Returns
    -------
    float
        ELBO value
    """

    # Sample time step t with importance sampling
    if not hasattr(model, "importance_weights"):
        alpha_bar_prev = torch.cat([torch.tensor([1.0]).to(model.alpha_bar.device), model.alpha_bar[:-1]])
        model.importance_weights = (alpha_bar_prev - model.alpha_bar) / (model.alpha * (1 - model.alpha_bar))
        model.importance_weights /= model.importance_weights.sum()
    
    t = torch.multinomial(model.importance_weights, num_samples=x0.shape[0], replacement=True).unsqueeze(-1).to(x0.device)

    epsilon = torch.randn_like(x0)

    # Forward diffusion to produce image at step t
    xt = model.forward_diffusion(x0, t, epsilon)

    losses = nn.MSELoss(reduction='none')(epsilon, model.network(xt, t)).mean(dim=list(range(1, len(x0.shape))))

    # Calculate the importance weights for each sample
    sample_weights = 1 / (model.T * model.importance_weights[t].squeeze())

    # Update loss history and compute moving average
    if len(model.loss_history) == model.loss_history_size:
        model.loss_history.pop(0)
    model.loss_history.append(losses.detach().cpu())

    if len(model.loss_history) > 1:
        avg_losses = torch.stack(model.loss_history).mean(dim=0)
        # Normalize and invert to get new importance weights
        new_importance_weights = 1 / avg_losses
        new_importance_weights /= new_importance_weights.sum()
        model.importance_weights = new_importance_weights.to(model.importance_weights.device)

    return -(losses * sample_weights).mean()