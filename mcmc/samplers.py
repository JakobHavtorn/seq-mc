import torch
import tqdm
import pandas as pd


def inverse_transform_sampling(inverse_cdf, n_samples=1):
    """Use the inverse cumulative density function (CDF) to sample from the probability density function (PDF)"""
    u = torch.rand(n_samples)
    return inverse_cdf(u)


def rejection_sampling(n_samples, proposal_dist, target_dist_log_prob, M):
    """Use a proposal distribution that is proportional to and covers the target distribution to sample the target.

    A proportionality costant M must be provided, such that p(theta|y) / g(theta) <= M for all theta where p is the
    target distribution and g is the proposal distribution (potentially unnormalized).
    """
    estimand_names = ["log_prob", "failed_attempts"]
    estimands = []
    samples = []
    M = torch.log(M)
    failed_attempts = 0
    while len(samples) < n_samples:
        proposal = proposal_dist.sample()

        target_log_prob = target_dist_log_prob(proposal)
        proposal_log_prob = proposal_dist.log_prob(proposal)

        acceptance = target_log_prob - (M + proposal_log_prob)

        event = torch.rand_like(acceptance).log()
        if acceptance > event:
            samples.append(proposal)
            estimands.append(torch.tensor([target_log_prob.item(), failed_attempts]))
            failed_attempts = 0
        else:
            failed_attempts += 1

    samples = torch.stack(samples)
    estimands = torch.stack(estimands)
    if samples.ndim == 1:
        samples = samples.view(-1, 1)
    estimands = torch.cat([estimands, samples], axis=-1)
    thetas = [f"theta_{i}" for i in range(samples.shape[-1])]
    estimand_names.extend(thetas)
    return samples, (estimand_names, estimands)


def metropolis(initial_point, n_samples, proposal_dist, target_dist_log_prob):
    states = [initial_point]
    current = initial_point
    accept_reject = []

    for i in range(n_samples):
        proposal = current + proposal_dist.sample()

        prop_log_prob = target_dist_log_prob(proposal)
        curr_log_prob = target_dist_log_prob(current)

        acceptance = prop_log_prob - curr_log_prob
        event = torch.rand(1).log()
        if acceptance > event:
            current = proposal

        states.append(current)
        accept_reject.append(float(acceptance > event))

    samples = torch.stack(states)
    return samples, accept_reject


def mh_correction(current, proposal, proposal_dist):
    proposal_relative = current - proposal + proposal_dist.mean
    current_relative = proposal - current + proposal_dist.mean
    proposal_prob = proposal_dist.log_prob(proposal_relative)
    current_prob = proposal_dist.log_prob(current_relative)
    return proposal_prob - current_prob


def metropolis_hastings(initial_point, n_samples, proposal_dist, target_dist_log_prob):
    states = [initial_point]
    current = initial_point
    proposal = None
    accept_reject = []

    assert not proposal_dist.log_prob(current).isnan(), "Initial point has NaN log_prob under proposal distribution"
    assert not target_dist_log_prob(current).isnan(), "Initial point has NaN log_prob under target distribution"

    for i in range(n_samples):
        # Since the proposal distribution is not symmetric, we need to offset the samples by the mean to not drift
        previous_proposal = proposal.clone() if proposal is not None else initial_point
        proposal = current + proposal_dist.sample() - proposal_dist.mean
        correction = mh_correction(current, proposal, proposal_dist)

        prop_log_prob = target_dist_log_prob(proposal)
        curr_log_prob = target_dist_log_prob(current)

        acceptance = prop_log_prob - curr_log_prob + correction
        event = torch.rand(1).log()
        if acceptance > event:
            current = proposal
        accept_reject.append(float(acceptance > event))

        states.append(current)

    samples = torch.stack(states)
    return samples, accept_reject


def metropolis_hastings_batched(initial_point, n_samples, proposal_dist, target_dist_log_prob):
    """Metropolis-Hastings sampling of as many parallel chains as given initial points.

    Multidimensional (dimension > 1) Metropolis-Hastings is theoretically optimal at 23% acceptance rate [1]

    Args:
        initial_point (torch.Tensor): Shape [n_chains, dimension]

    Returns:
        samples (torch.Tensor): Samples of shape [n_samples, n_chains, dimension]

    [1] Bayesian Data Analysis 3, Andrew Gelman, John B. Carlin, Aki Vehtari, Donald B. Rubin, Hal S. Stern,
        David B. Dunson, 1995, 3rd edition, http://www.stat.columbia.edu/~gelman/book/.
    """
    if initial_point.ndim == 1:
        n_chains = torch.Size([1])
        initial_point = initial_point.view(*n_chains, *initial_point.shape)
    else:
        n_chains = initial_point.shape[:1]

    estimand_names = ["log_prob", "acceptance"]
    estimands = []
    states = [initial_point]
    current = initial_point
    current_log_prob = target_dist_log_prob(initial_point)
    proposal = None
    accept_reject = []

    assert (
        not proposal_dist.log_prob(current).isnan().any()
    ), f"Initial point has NaN log_prob under proposal distribution {current}"
    assert (
        not target_dist_log_prob(current).isnan().any()
    ), f"Initial point has NaN log_prob under target distribution {current}"

    for i in range(n_samples):
        # Since the proposal distribution is not symmetric, we need to offset the samples by the mean to not be directionally biased
        proposal = current + proposal_dist.sample(n_chains) - proposal_dist.mean
        correction = mh_correction(current, proposal, proposal_dist)

        curr_log_prob = target_dist_log_prob(current)
        prop_log_prob = target_dist_log_prob(proposal)

        acceptance = prop_log_prob - curr_log_prob + correction
        event = torch.rand_like(acceptance).log()
        accepted = acceptance > event
        current[accepted] = proposal[accepted]  # Update only for the accepted proposals
        current_log_prob[accepted] = prop_log_prob[accepted]

        states.append(current.clone())

        accept_reject.extend(list((accepted).type(torch.FloatTensor)))

        estimands.append(torch.stack([current_log_prob, accepted.type(torch.FloatTensor)]).permute(1, 0))

    samples = torch.stack(states[1:])  # Exclude initial point
    estimands = torch.stack(estimands)
    if samples.ndim == 1:
        samples = samples.view(-1, 1)
    estimands = torch.cat([estimands, samples], axis=-1)
    thetas = [f"theta_{i}" for i in range(samples.shape[-1])]
    estimand_names.extend(thetas)
    return samples, (estimand_names, estimands)


def compute_log_prob_gradient(proposal_sample, target_dist_log_prob):
    """Computes and returns the gradient of a proposal sample under a log target distribution

    We set the proposal to require gradients, zero the gradient and then compute the gradient.
    In the computation we sum over the batch examples (should there be any).
    Then we reset the `requires_grad` to `False` to enable later in-place operations on the proposal.
    """
    proposal_sample.requires_grad_(True)
    proposal_sample.grad = None
    target_dist_log_prob(proposal_sample).sum().backward()
    proposal_sample.requires_grad_(False)
    return proposal_sample.grad


def hamiltonian_monte_carlo(
    initial_point,
    n_samples,
    momentum_dist,
    target_dist_log_prob,
    leapfrog_steps=10,
    leapfrog_stepsize=0.1,
    return_full_path=False,
):
    accept_reject = []
    full_hmc_path = []
    samples = [initial_point]
    M_inv = 1 / momentum_dist.variance

    for hmc_step in range(n_samples):
        # Sample momentum and increment current/proposal
        current_momentum = momentum_dist.sample()
        proposal_momentum = current_momentum.clone()  # Clone since we will modify in-place

        current_sample = samples[-1].clone()  # Clone since we will modify in-place
        proposal_sample = current_sample.clone()  # Clone since we will modify in-place

        # Leapfrog
        leapfrog_samples = []
        grad = compute_log_prob_gradient(proposal_sample, target_dist_log_prob)
        proposal_momentum += 0.5 * leapfrog_stepsize * grad

        for step in range(leapfrog_steps - 1):
            proposal_sample += leapfrog_stepsize * M_inv * proposal_momentum
            grad = compute_log_prob_gradient(proposal_sample, target_dist_log_prob)
            proposal_momentum += leapfrog_stepsize * grad
            leapfrog_samples.append(proposal_sample.clone())

        proposal_sample += leapfrog_stepsize * M_inv * proposal_momentum
        grad = compute_log_prob_gradient(proposal_sample, target_dist_log_prob)
        proposal_momentum += 0.5 * leapfrog_stepsize * grad
        leapfrog_samples.append(proposal_sample.clone())

        # Invert momentum for reversibility
        proposal_momentum = -proposal_momentum

        # Metropolis acceptance
        proposal_sample_log_prob = target_dist_log_prob(proposal_sample)
        current_sample_log_prob = target_dist_log_prob(current_sample)

        proposal_momentum_log_prob = momentum_dist.log_prob(proposal_momentum)
        current_momentum_log_prob = momentum_dist.log_prob(current_momentum)

        target = proposal_sample_log_prob - current_sample_log_prob
        adjustment = proposal_momentum_log_prob - current_momentum_log_prob
        acceptance = target + adjustment

        event = torch.rand_like(acceptance).log()
        if acceptance > event:
            samples.append(proposal_sample)
            if return_full_path:
                full_hmc_path.append(leapfrog_samples)
        else:
            samples.append(current_sample)
        accept_reject.append(float(acceptance > event))

    samples = torch.stack(samples)
    if return_full_path:
        full_hmc_path = torch.stack([torch.stack(paths) for paths in full_hmc_path])  # List of list of tensor to tensor
        return samples, accept_reject, full_hmc_path
    return samples, accept_reject


def hamiltonian_monte_carlo_batched(
    initial_point,
    n_samples,
    momentum_dist,
    target_dist_log_prob,
    leapfrog_steps=10,
    leapfrog_stepsize=0.1,
    return_full_path=False,
    use_progressbar=False,
):
    """
    Args:
        initial_point (torch.Tensor): Shape [n_chains, dimension] or [dimension] where we we make it [1, dimension]
        n_samples (int): Number of samples to return
        momentum_dist (torch.distributions.Distribution): Distribution for the momentum
        target_dist_log_prob (callable): Callable likelihood function taking only one argument of same shape as `initial_point`
        leapfrog_steps (int): Number of steps for Leapfrog algorithm (integrator)
        leapfrog_stepsize (float): Stepsize of the Leapfrog algorithm. TODO This could be made torch.Tensor of shape [n_chains] to allow per chain tuning
        return_full_path (bool): In addtion to the samples and estimands, return also every step of Leapfrog. Defaults to False.

    Returns:
        samples (torch.Tensor): HMC samples of shape [n_samples // 2, n_chains, dimensionality] `(shuffle(chains[n_samples//2:]))`
        chains (torch.Tensor): All samples of the HMC chain including burnin [n_samples, n_chains, dimensionality]
        estimand_names (list): List of string of the names of the estimands ['log_prob', 'acceptance', 'theta0', ...]
        estimands (torch.Tensor): Values of the estimands for all samples and chains [n_samples, n_chains, n_estimands]
        full_hmc_path (torch.Tensor): Full path of HMC of shape [n_samples, leapfrog_steps, n_chains, dimensionality]
    """
    if initial_point.ndim == 1:
        # Add chains dimension
        n_chains = torch.Size([1])
        initial_point = initial_point.view(*n_chains, *initial_point.shape)
    elif initial_point.ndim == 2:
        n_chains = initial_point.shape[:1]
    else:
        raise ValueError(f"Dimensionality of `initial_point` must be 2 but got {initial_point.ndim}")

    estimand_names = ["log_prob", "acceptance"]
    estimands = []
    current_sample_log_prob = target_dist_log_prob(initial_point)
    full_hmc_path = []
    samples = [initial_point]
    M_inv = 1 / momentum_dist.variance

    iterator = range(n_samples) if not use_progressbar else tqdm.tqdm(range(n_samples))
    for hmc_step in iterator:
        # Sample momentum and increment current/proposal
        current_momentum = momentum_dist.sample(n_chains)
        proposal_momentum = current_momentum.clone()  # Clone since we will modify in-place

        current_sample = samples[-1].clone()  # Clone since we will modify in-place
        proposal_sample = current_sample.clone()  # Clone since we will modify in-place

        # Leapfrog
        leapfrog_samples = []
        grad = compute_log_prob_gradient(proposal_sample, target_dist_log_prob)
        proposal_momentum += 0.5 * leapfrog_stepsize * grad

        for step in range(leapfrog_steps - 1):
            proposal_sample += leapfrog_stepsize * M_inv * proposal_momentum
            grad = compute_log_prob_gradient(proposal_sample, target_dist_log_prob)
            proposal_momentum += leapfrog_stepsize * grad
            leapfrog_samples.append(proposal_sample.clone())

        proposal_sample += leapfrog_stepsize * M_inv * proposal_momentum
        grad = compute_log_prob_gradient(proposal_sample, target_dist_log_prob)
        proposal_momentum += 0.5 * leapfrog_stepsize * grad
        leapfrog_samples.append(proposal_sample.clone())

        # Invert momentum for reversibility
        proposal_momentum = -proposal_momentum

        # Metropolis acceptance
        proposal_sample_log_prob = target_dist_log_prob(proposal_sample)
        current_sample_log_prob = target_dist_log_prob(current_sample)

        proposal_momentum_log_prob = momentum_dist.log_prob(proposal_momentum)
        current_momentum_log_prob = momentum_dist.log_prob(current_momentum)

        target = proposal_sample_log_prob - current_sample_log_prob
        adjustment = proposal_momentum_log_prob - current_momentum_log_prob
        acceptance = target + adjustment

        event = torch.rand_like(acceptance).log()
        accepted = acceptance > event
        # print(accepted.sum())
        current_sample[accepted] = proposal_sample[accepted]  # Update only for the accepted proposals
        current_sample_log_prob[accepted] = proposal_sample_log_prob[accepted]

        samples.append(current_sample.clone())

        estimands.append(torch.stack([current_sample_log_prob, accepted.float()]).permute(1, 0))

        if return_full_path:
            full_hmc_path.append(leapfrog_samples)

    samples = torch.stack(samples[1:])  # Exclude initial point
    estimands = torch.stack(estimands)
    if samples.ndim == 1:
        samples = samples.view(-1, 1)
    estimands = torch.cat([estimands, samples], axis=-1)
    thetas = [f"theta_{i}" for i in range(samples.shape[-1])]
    estimand_names.extend(thetas)

    chains = samples.clone()
    samples = samples[n_samples // 2 :].flatten(start_dim=0, end_dim=1)
    samples = samples[torch.randperm(samples.size()[0])]

    if return_full_path:
        full_hmc_path = torch.stack([torch.stack(paths) for paths in full_hmc_path])  # List of list of tensor to tensor
        return samples, chains, estimand_names, estimands, full_hmc_path
    return samples, chains, estimand_names, estimands
