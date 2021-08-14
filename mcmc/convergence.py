import torch
import pandas as pd


def convergence_test(names, estimands):
    """Test the convergence of a set of MCMC chains on a number of estimands as suggested in [1].

    Args:
        estimands (torch.Tensor): MCMC chains of shape [n_samples, n_chains, n_estimands]

    Returns:
        tuple: DataFrame(R_hat, n_eff, varplus_estimand), variogram, autocorrelations, T

    [1] Bayesian Data Analysis 3, Andrew Gelman, John B. Carlin, Aki Vehtari, Donald B. Rubin, Hal S. Stern,
        David B. Dunson, 1995, 3rd edition, http://www.stat.columbia.edu/~gelman/book/.
    """
    n_samples = estimands.shape[0]
    n_chains = estimands.shape[1]

    split_estimands = split_chains(estimands, n_splits=2)
    varplus_estimand, between_chain_variance, within_chain_variance = compute_variance_estimate(split_estimands)
    R_hat = compute_scale_reduction_factor(varplus_estimand, within_chain_variance)
    variogram = compute_variogram(split_estimands)
    autocorrelations = compute_autocorrelation(variogram, varplus_estimand)
    n_eff, T = estimate_effective_n_samples(autocorrelations, n_chains=n_chains, n_samples=n_samples)

    R_hat, n_eff, varplus_estimand, T = R_hat.numpy(), n_eff.numpy(), varplus_estimand.numpy(), T.numpy()
    metrics = {name: [R_hat[i], n_eff[i], varplus_estimand[i], T[i]] for i, name in enumerate(names)}
    metrics = pd.DataFrame(metrics)
    metrics.index = ["R_hat", "n_effective", "var_plus", "T"]
    return metrics, variogram.numpy(), autocorrelations.numpy(), T


def split_chains(samples, n_splits=2):
    """Splits an MCMC tensor of shape [n_samples, n_chains, n_estimands] into n_chains x n_splits chains"""
    n_samples = samples.shape[0]
    samples_per_split_chain = n_samples // n_splits
    # Remove any extra number of sample < n_splits from the left (initial point etc.)
    samples = samples[-samples_per_split_chain * n_splits :]
    new_chains = torch.split(samples, samples_per_split_chain)
    samples = torch.cat(new_chains, axis=1)
    return samples


def compute_variance_estimate(split_estimands):
    """Estimates the posterior marginal variance of each estimand in the split chains

    The variance estimate is an overestimate that approaches the true variance from above
    as n_samples goes to infinity and is unbiased.

    Args:
        split_estimands (torch.Tensor): [n_samples, n_chains, n_estimands]
    """
    n_samples = split_estimands.shape[0]
    n_chains = split_estimands.shape[1]

    chain_averages = split_estimands.mean(axis=0)  # [n_chains, n_estimands]
    total_averages = split_estimands.mean(axis=[0, 1])  # [n_estimands]
    between_chain_variance = n_samples / (n_chains - 1) * (chain_averages - total_averages).pow(2).sum(axis=0)

    # Broadcast subtraction to chain n_estimands by making it leftmost, and then put it back
    diff = (split_estimands - chain_averages).pow(2).sum(axis=0)  # []
    s_j = 1 / (n_samples - 1) * diff
    within_chain_variance = 1 / n_chains * s_j.sum(axis=0)

    varplus_estimand = (n_samples - 1) / n_samples * within_chain_variance + 1 / n_samples * between_chain_variance
    return varplus_estimand, between_chain_variance, within_chain_variance


def compute_scale_reduction_factor(varplus_estimand, within_chain_variance):
    """Estimate the factor by which the scale of the current distribution of the estimands might be reduced
    if the simulations were continued in the limit n -> infinity
    """
    R_hat = torch.sqrt(varplus_estimand / within_chain_variance)
    return R_hat


def compute_variogram(estimands):
    """Compute the variogram used for computing the autocorrelation estimate"""
    n_samples = estimands.shape[0]
    n_chains = estimands.shape[1]

    variogram = []
    for t in range(1, n_samples):
        factor = 1 / (n_chains * (n_samples - t))
        lag_t_diff = (estimands[t:, ...] - estimands[:-t, ...]).pow(2)
        lag_t_diff = lag_t_diff.sum(axis=0).sum(axis=0)
        variogram_t = factor * lag_t_diff
        variogram.append(variogram_t)
    variogram = torch.stack(variogram)  # [n_samples - 1, n_estimands]
    return variogram


def compute_autocorrelation(variogram, varplus_estimand):
    """Estimate the autocorrelation of the estimands"""
    autocorrelations = 1 - variogram / (2 * varplus_estimand)
    return autocorrelations


def estimate_effective_n_samples(autocorrelations, n_chains, n_samples=None):
    """Estimate the effective number of samples obtained from an MCMC chain

    Args:
        autocorrelations (torch.Tensor): Tensor of shape [n_samples - 1, n_estimands]

    Returns:
        n_eff (torch.Tesnor): Tensor with the effective number of samples per estimand [n_estimands]
        T (int): The last index into `autocorrelations` before they turn negative (indicating too high variance)
    """
    if n_samples is None:
        n_samples = autocorrelations.shape[0] + 1

    # Compute first time the sum of autocorrelation estimates for two successive lags \rho_{2t} + \rho_{2t+1} is negative.
    # Use this time index as the limit of the partial sum of autocorrelations (to not include high-variance autocorrelation estimates)
    diff = autocorrelations[:-1] + autocorrelations[1:]
    diff = diff[::2]

    any_nonneg, first_nonneg = first_nonnegative(-diff, axis=0)  # First nonzero elements
    T = 2 * first_nonneg  # Make up for the subsampling in diff above
    T[~any_nonneg] = len(autocorrelations) - 1  # Use full sequence for those without zero crossing

    # Compute the sum of autocorrelations[:T] for each estimands's T_i in turn (couldn't figure out how to vectorize)
    sums = []
    for i_estimand in range(autocorrelations.shape[1]):
        sums.append(autocorrelations[: T[i_estimand], i_estimand].sum(axis=0))
    sums = torch.tensor(sums)

    n_eff = (n_samples * n_chains) / (1 + 2 * sums)
    return n_eff, T


def first_nonnegative(tensor, axis=0):
    """Returns the index of the first nonnegative element in the tensor and if any elements where nonnegative.

    An element is the first nonzero element if it is nonzero and the cumulative sum of a nonzero indicator is 1.

    If there are no nonnegative elements, this method returns value_for_all_nonnegative (-1 by default).

    Example:
        x = torch.randn(5, 7)
        x[3, 3] = 0
        any_nonneg, idx_first_nonneg = first_nonnegative(x)

    Args:
        x (torch.Tensor): Tensor of some shape
        axis (int): The axis along which to operate

    Returns:
        any_nonneg (torch.Tensor): Boolean array `True` where at least one element was nonnegative.
        idx_first_nonneg (torch.Tensor): Integer array of indices where the first nonnegative element is.
                                         Equal to -1 where no nonnegative elements where found.
    """
    nonneg = tensor > 0
    any_nonneg, idx_first_nonneg = ((nonneg.cumsum(axis) == 1) & nonneg).max(axis)
    idx_first_nonneg[~any_nonneg] = -1
    return any_nonneg, idx_first_nonneg


def estimands_to_dataframe(names, estimands):
    """Convert the estimand names and the estimand tensor of shape [n_samples, n_chains, n_estimands] to a dataframe.

    To summarize the MCMC chain call:
        series.describe().T
        series.quantile([0.05, 0.95]).T  # 95% posterior intervals

    Args:
        names (list): Names of the estimands
        estimands (torch.Tensor): Estimands of the chains per sample shaped as [n_samples, n_chains, n_estimands]

    Returns:
        pd.DataFrame: With all estimands flattened to a long list [n_samples * n_chains] and n_estimands columns.
    """
    data = {name: estimands[..., i].flatten() for i, name in enumerate(names)}
    series = pd.DataFrame(data)
    return series


def create_estimates_report(names, estimands):
    """Create a DataFrame summarizing the estimates of the MCMC chain.

    Args:
        names (list): Names of the estimands
        estimands (torch.Tensor): Estimands of the chains per sample shaped as [n_samples, n_chains, n_estimands]

    Returns:
        pd.DataFrame: With count, mean, std, min, max and quanties
    """
    df = estimands_to_dataframe(names, estimands)
    quantiles = df.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    statistics = pd.DataFrame(dict(count=df.count(), mean=df.mean(), std=df.std(), min=df.min(), max=df.max()))
    report = statistics.join(quantiles)
    return report
