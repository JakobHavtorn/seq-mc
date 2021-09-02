# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from types import SimpleNamespace

import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

from tqdm import tqdm



# %% [markdown]
# # Problem 3: Parameter estimation in the stochastic volatility model [11p]
# %% [markdown]
# ## a) Grid search for phi

# %%
def bootstrap_pf_stochastic_volatility(initial_particles, phi, sigma, beta, verbose=True, seed=0):
    if seed is not None:
        np.random.seed(seed)

    N = len(initial_particles)
    if verbose:
        print(f"Running with {N} particles")

    loglikelihood = 0
    weights = [np.array([1/N] * N)] + [None] * T
    particles = [initial_particles] + [None] * T  # draw initial particles
    mean_observation = [None] * T
    prediction = [None] * T
    marginal_filtering = [None] * T

    for t in range(T):
        # RESAMPLE
        ancestor_indices = np.random.choice(range(N), p=weights[t], replace=True, size=N)

        # PROPAGATE
        # state
        proposal_dist = scipy.stats.norm(phi * particles[t][ancestor_indices], sigma)
        particles[t+1] = proposal_dist.rvs()

        # measurement
        measurement_dist = scipy.stats.norm(0, np.sqrt(beta ** 2 * np.exp(particles[t+1])))
        # mean observation
        mean_observation[t] = scipy.stats.norm(0, np.sqrt(beta ** 2 * np.exp(np.mean(particles[t+1])))).rvs()

        # WEIGHT
        log_weights_unnorm = measurement_dist.logpdf(observation_data[t])
        weights_unnorm = np.exp(log_weights_unnorm - np.max(log_weights_unnorm))
        weights[t+1] = weights_unnorm / np.sum(weights_unnorm)

        prediction[t] = np.mean(particles[t])
        marginal_filtering[t] = np.sum(weights[t] * particles[t])

        loglikelihood += np.log(np.sum(weights_unnorm)) - np.log(N) + np.max(log_weights_unnorm)

    particles = np.array(particles[:-1])  # remove initial state
    marginal_filtering = np.array(marginal_filtering)
    mean_observation = np.array(mean_observation)
    loglikelihood = np.array(loglikelihood)

    output = SimpleNamespace(
        particles=particles,
        marginal_filtering=marginal_filtering,
        mean_observation=mean_observation,
        loglikelihood=loglikelihood,
    )
    return output


# %%
observation_data = pd.read_csv("./seOMXlogreturns2012to2014.csv")
T = observation_data.shape[0]
observation_data = observation_data.to_numpy()[:, 0]
observation_data.shape


# %%
# phis = np.linspace(0.1, 1, 10)  # 0.98
phis = np.linspace(0.95, 1, 11)  # 0.98
sigma = 0.16
beta = 0.70

N = 500
initial_particle_dist = scipy.stats.norm(0, 1)

phis


# %%
# Bootstrap Particle Filter parameter estimation via grid

np.random.seed(0)

loglikelihood = []

for phi in tqdm(phis):
    loglikelihood_ = []

    for repeat in range(10):

        initial_particles = initial_particle_dist.rvs(N)

        output = bootstrap_pf_stochastic_volatility(initial_particles, phi, sigma, beta, verbose=False, seed=None)
        
        loglikelihood_.append(output.loglikelihood)

    loglikelihood.append(loglikelihood_)

loglikelihood = np.array(loglikelihood)


# %%
loglikelihood.shape


# %%
np.mean(loglikelihood, axis=1)


# %%
fig, ax = plt.subplots(1, 1)
ax.boxplot(loglikelihood.T)
ax.set_xticklabels(phis.round(3))
ax.set_xlabel("$\phi$")
ax.set_ylabel("log-likelihood")
plt.savefig(f"./figures/phis_loglikelihood_bpf_{min(phis)}_{max(phis)}.pdf", bbox_inches="tight")
plt.close()

# %% [markdown]
# ## b) Particle Metropolis Hastings for parameter estimation

# %%
def mh_correction(current, proposal, proposal_dist):
    proposal_relative = current - proposal + proposal_dist.mean()
    current_relative = proposal - current + proposal_dist.mean()
    proposal_prob = proposal_dist.logpdf(proposal_relative).sum()  # Sum over dim of parameters
    current_prob = proposal_dist.logpdf(current_relative).sum()  # Sum over dim of parameters
    return proposal_prob - current_prob


def particle_metropolis_hastings(n_steps, initial_param, param_random_walk_proposal, param_prior_logpdf, initial_particle_dist, n_particles, phi, verbose=0, seed=0):
    if seed is not None:
        np.random.seed(seed)

    current_param = initial_param
    initial_particles = initial_particle_dist.rvs(n_particles)
    output = bootstrap_pf_stochastic_volatility(initial_particles, phi, sigma=current_param[0], beta=current_param[1], verbose=verbose>1, seed=None)
    current_loglikelihood = output.loglikelihood

    params = []
    loglikelihoods = []
    for m in range(n_steps):
        proposed_param = current_param + param_random_walk_proposal.rvs() - param_random_walk_proposal.mean()

        if proposed_param[0] < 0 or proposed_param[1] < 0:
            # if the proposed parameters are out of domain, we perform the Metropolis rejection already here.
            # if verbose > 1:
            print(f"Rejected run {m} due to domain error in the proposed parameters")

            params.append(current_param)
            loglikelihoods.append(current_loglikelihood)
            continue

        initial_particles = initial_particle_dist.rvs(n_particles)
        output = bootstrap_pf_stochastic_volatility(initial_particles, phi, sigma=proposed_param[0], beta=proposed_param[1], verbose=verbose>1, seed=None)
        proposed_loglikelihood = output.loglikelihood

        correction = mh_correction(current_param, proposed_param, param_random_walk_proposal)

        proposed_param_logprob = param_prior_logpdf(proposed_param ** 2)  # square since prior is over sigma**2 and beta**2
        current_param_logprob = param_prior_logpdf(current_param ** 2)
        
        acceptance = proposed_param_logprob - current_param_logprob + proposed_loglikelihood - current_loglikelihood + correction
        event = np.log(np.random.uniform(0, 1))
        if acceptance > event:
            current_param = proposed_param
            current_loglikelihood = proposed_loglikelihood

        params.append(current_param)
        loglikelihoods.append(current_loglikelihood)
        
        if verbose:
            l = len(loglikelihoods)
            accept_rate = round(len(np.unique(loglikelihoods[l//2:])) / len(loglikelihoods[l//2:]), 3) * 100
            print(f"{m}/{n_steps} | acc_prob={np.exp(acceptance)*100:4.1f}, acc_rate={accept_rate:4.1f}, current_param={list(current_param)}")
        
    return np.array(params), np.array(loglikelihoods)


# %%
def param_prior_logpdf(params):
    d = scipy.stats.invgamma(a=0.01, scale=0.01)
    return d.logpdf(params[0]) + d.logpdf(params[1])


# %%
def param_prior_pdf(params):
    d = scipy.stats.invgamma(a=0.01, scale=0.01)
    return np.exp(d.logpdf(params[0]) + d.logpdf(params[1]))


# %%
# def param_prior_logpdf(params):
#     d = scipy.stats.uniform(0, 1)
#     return d.logpdf(params[0]) + d.logpdf(params[1])


# %%
param_rw_proposal = scipy.stats.norm([0, 0], 0.1)


# %%
param_rw_proposal.rvs()


# %%
# Estimate parameters (infer posterior p(sigma,beta|y_data)) using Particle Metropolis Hastings
# M=30 sigma=0.05 ar=0.02
# M=30 sigma=0.01 ar=0.52
param_rw_proposal = scipy.stats.norm([0, 0], 0.1)
initial_particle_dist = scipy.stats.norm(0, 1)
initial_param = np.array([0.5, 0.5])
phi = 0.985
N = 500  # Number of APF particles
M = 60000  # Number of PMH runs


# %%
params, loglikelihoods = particle_metropolis_hastings(M, initial_param, param_rw_proposal, param_prior_logpdf, initial_particle_dist, N, phi, verbose=1, seed=0)
len(params), len(loglikelihoods), len(np.unique(loglikelihoods)), round(len(np.unique(loglikelihoods[M//2:])) / len(loglikelihoods[M//2:]), 2)


# %%
plt.hist(loglikelihoods, bins=50, density=True)
plt.xlabel("Log-Likelihood")
plt.ylabel("Density")
plt.savefig(f"./figures/sigma_beta_loglikelihoods_{N}_particles_{M}_pmh_iterations.pdf", bbox_inches="tight")
plt.close()


# %%
params[np.argmax(loglikelihoods)], np.argmax(loglikelihoods)


# %%
plt.hist(params[:,0], bins=50, density=True, label="Estimated posterior")
plt.plot([0.16, 0.16], [0, plt.gca().get_ylim()[1]], 'r', label='True value')
plt.xlabel("sigma")
plt.ylabel("Density")
plt.legend()
plt.savefig(f"./figures/marginal_posterior_sigma_{N}_particles_{M}_pmh_iterations.pdf", bbox_inches="tight")
plt.close()


# %%
plt.hist(params[:,1], bins=50, density=True, label="Estimated posterior")
plt.plot([0.70, 0.70], [0, plt.gca().get_ylim()[1]], 'r', label='True value')
plt.xlabel("beta")
plt.ylabel("Density")
plt.legend()
plt.savefig(f"./figures/marginal_posterior_beta_{N}_particles_{M}_pmh_iterations.pdf", bbox_inches="tight")
plt.close()


# %%
plt.scatter(params[:,0], loglikelihoods);
plt.close()


# %%
plt.scatter(params[:,1], loglikelihoods);
plt.close()
