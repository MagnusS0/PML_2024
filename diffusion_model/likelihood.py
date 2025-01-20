import torch
import numpy as np
from scipy import integrate

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.

    Adopted directly from:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.

    Adopted directly from:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).

    Adopted directly from:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

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


def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and 
      standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

def ode_likelihood(x, 
                   score_model,
                   marginal_prob_std, 
                   diffusion_coeff,
                   batch_size=64, 
                   device='cuda',
                   eps=1e-5):
  """Compute the likelihood with probability flow ODE.
  
  Args:
    x: Input data.
    score_model: A PyTorch model representing the score-based model.
    marginal_prob_std: A function that gives the standard deviation of the 
      perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the 
      forward SDE.
    batch_size: The batch size. Equals to the leading dimension of `x`.
    device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
    eps: A `float` number. The smallest time step for numerical stability.

  Returns:
    z: The latent code for `x`.
    bpd: The log-likelihoods in bits/dim.
  """

  # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
  epsilon = torch.randn_like(x)
      
  def divergence_eval(sample, time_steps, epsilon):      
    """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    with torch.enable_grad():
      sample.requires_grad_(True)
      score_e = torch.sum(score_model(sample, time_steps) * epsilon)
      grad_score_e = torch.autograd.grad(score_e, sample)[0]
    return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    
  
  shape = x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper for evaluating the score-based model for the black-box ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def divergence_eval_wrapper(sample, time_steps):
    """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
    with torch.no_grad():
      # Obtain x(t) by solving the probability flow ODE.
      sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
      time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
      # Compute likelihood.
      div = divergence_eval(sample, time_steps, epsilon)
      return div.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):
    """The ODE function for the black-box solver."""
    time_steps = np.ones((shape[0],)) * t    
    sample = x[:-shape[0]]
    logp = x[-shape[0]:]
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
    logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
    return np.concatenate([sample_grad, logp_grad], axis=0)

  init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
  # Black-box ODE solver
  res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
  zp = torch.tensor(res.y[:, -1], device=device)
  z = zp[:-shape[0]].reshape(shape)
  delta_logp = zp[-shape[0]:].reshape(shape[0])
  sigma_max = marginal_prob_std(1.)
  prior_logp = prior_likelihood(z, sigma_max)
  bpd = -(prior_logp + delta_logp) / np.log(2)
  N = np.prod(shape[1:])
  bpd = bpd / N + 8.
  return z, bpd