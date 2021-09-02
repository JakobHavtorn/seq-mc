# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from types import SimpleNamespace

import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

from tqdm import tqdm

# %% [markdown]
# # Problem 3: Parameter estimation in the stochastic volatility model [11p]

# %%
def backtrack_genealogy(list_index, list_sample):
    """Requires initial particle to be at list_sample[-1] and len(list_sample) = len(list_index) + 1"""
    aux_list_index = copy.deepcopy(list_index)
    genealogy = [list_sample[-2].reshape(1, -1)]  # list_sample[-2] is the last particle

    T = len(list_index)
    for t in range(T - 1, -1, -1):  # [4, 3, 2, 1, 0]
        genealogy.insert(0, list_sample[t-1][aux_list_index[t]].reshape(1, -1))
        aux_list_index[t-1] = aux_list_index[t-1][aux_list_index[t]]

    genealogy = np.concatenate(genealogy, axis=0)  # (x_0, x_1, x_2, ..., x_T)
    return genealogy

# %%
observation_data = pd.read_csv("./seOMXlogreturns2012to2014.csv")
T = observation_data.shape[0]
observation_data = observation_data.to_numpy()[:, 0]
observation_data.shape, T



# %% [markdown]
# ## c) Particle Gibbs for parameter estimation

# %%
def bootstrap_pf_gibbs_stochastic_volatility(N, initial_particle_dist, reference_trajectory, phi, sigma, beta, observation_data, verbose=True, seed=None):
    """BPF as a Particle Gibs Kernel.

    Args:
        initial_particles: (N,)
        reference_trajectory: (T,) with initial value at index -1
        phi ([type]): parameter
        sigma ([type]): parameter
        beta ([type]): parameter
        verbose (bool, optional): Print statuses. Defaults to True.
        seed (int, optional): Seed. Defaults to 0.
    """
    if seed is not None:
        np.random.seed(seed)
    if verbose:
        print(f"Running with {N} particles")

    initial_particles = initial_particle_dist.rvs(N)

    # deterministically set reference trajectory
    initial_particles[-1] = reference_trajectory[-1]  # Condition on the reference trajectory

    loglikelihood = 0
    weights = [None] * T + [np.array([1/N] * N)]
    particles = [None] * T + [initial_particles]  # draw initial particles
    mean_observation = [None] * T
    prediction = [None] * T
    marginal_filtering = [None] * T
    ancestor_indices = [None] * T

    for t in range(T):
        # RESAMPLE
        a_indices = np.random.choice(range(N), p=weights[t-1], replace=True, size=N)
        ancestor_indices[t] = a_indices

        # PROPAGATE
        # state
        proposal_dist = scipy.stats.norm(phi * particles[t-1][a_indices], sigma)
        particles[t] = proposal_dist.rvs()
        
        # deterministically set reference trajectory
        particles[t][-1] = reference_trajectory[t]  # Condition on the reference trajectory
        ancestor_indices[t][-1] = N - 1  # Update according to reference trajectory

        # measurement
        measurement_dist = scipy.stats.norm(0, np.sqrt(beta ** 2 * np.exp(particles[t])))
        # mean observation
        mean_observation[t] = scipy.stats.norm(0, np.sqrt(beta ** 2 * np.exp(np.mean(particles[t])))).rvs()

        # WEIGHT
        log_weights_unnorm = measurement_dist.logpdf(observation_data[t])
        weights_unnorm = np.exp(log_weights_unnorm - np.max(log_weights_unnorm))
        weights[t] = weights_unnorm / np.sum(weights_unnorm)

        prediction[t] = np.mean(particles[t])
        marginal_filtering[t] = np.sum(weights[t] * particles[t])

        loglikelihood += np.log(np.sum(weights_unnorm)) - np.log(N) + np.max(log_weights_unnorm)

    genealogy = backtrack_genealogy(ancestor_indices, particles)
    j = np.random.choice(range(N), p=weights[T-1], replace=False, size=1)
    reference_trajectory = genealogy[:, j].reshape(-1)  # (T+1, N) -> (T+1,)
    reference_trajectory = np.concatenate([reference_trajectory[1:], reference_trajectory[:1]])  # put initial at end

    particles = np.array(particles[:-1])  # remove initial state
    marginal_filtering = np.array(marginal_filtering)
    mean_observation = np.array(mean_observation)
    loglikelihood = np.array(loglikelihood)
    ancestor_indices = np.array(ancestor_indices)

    output = SimpleNamespace(
        particles=particles,
        marginal_filtering=marginal_filtering,
        mean_observation=mean_observation,
        loglikelihood=loglikelihood,
        ancestor_indices=ancestor_indices,
        reference_trajectory=reference_trajectory
    )
    return output


# %%
def particle_gibbs_sampler(M, markov_kernel, sample_parameter_conditional_dist, initial_reference_trajectory, observation_data, burn_in=100, verbose=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if burn_in >= M:
        raise ValueError(f"We need burn_in < M")
    if burn_in == 0:
        burn_in = 1  # discard initial reference trajectory
    
    reference_trajectories = [initial_reference_trajectory] + [None] * M
    parameters = [None] + [None] * M
    loglikelihoods = [None] + [None] * M
    # iterator = tqdm(range(1, M)) if verbose else range(1, M)
    for m in range(1, M + 1):
        parameters[m] = sample_parameter_conditional_dist(reference_trajectories[m-1], observation_data)
        
        kernel_out = markov_kernel(
            reference_trajectory=reference_trajectories[m-1],
            sigma=parameters[m][0],
            beta=parameters[m][0],
        )
        reference_trajectories[m] = kernel_out.reference_trajectory
        
        loglikelihoods[m] = kernel_out.loglikelihood
        
        if verbose:
            print(f"{m:3d}/{M:3d} | {parameters[m]} | {loglikelihoods[m]} | {np.min(reference_trajectories[m])}, {np.max(reference_trajectories[m])}, {np.mean(reference_trajectories[m])}")

    reference_trajectories = reference_trajectories[burn_in:]
    parameters = parameters[burn_in:]
    loglikelihoods = loglikelihoods[burn_in:]

    parameters = np.stack([np.array(p) for p in parameters])
    reference_trajectories = np.stack(reference_trajectories)
    loglikelihoods = np.stack(loglikelihoods)

    return reference_trajectories, parameters, loglikelihoods


# %%
from functools import partial


# %%
def sample_parameter_conditional_dist(states, observation_data):
    a = 0.01
    b = 0.01

    a_new = a + T/2
    b_sigma = b + 1/2 * np.sum( (states[1:] - phi * states[:-1]) ** 2 )
    b_beta = b + 1/2 * np.sum( np.exp(-states[1:]) * observation_data ** 2 )

    sigma_cond = scipy.stats.invgamma(a=a_new, scale=b_sigma)
    beta_cond = scipy.stats.invgamma(a=a_new, scale=b_beta)
    
    return np.sqrt(sigma_cond.rvs()), np.sqrt(beta_cond.rvs())


N = 5000
initial_particle_dist = scipy.stats.norm(0, 1)
phi = 0.985

markov_kernel = partial(
    bootstrap_pf_gibbs_stochastic_volatility,
    N=N,
    initial_particle_dist=initial_particle_dist,
    phi=phi,
    observation_data=observation_data,
    verbose=False,
    seed=None,
)


# %%
for M in [3, 350, 20000, 1000, 60000]:
    print(f"Running {M=}")

    initial_reference_trajectory = scipy.stats.norm(0, 1).rvs(T+1)

    states, parameters, loglikelihoods = particle_gibbs_sampler(
        M,
        markov_kernel,
        sample_parameter_conditional_dist,
        initial_reference_trajectory,
        burn_in=100 if M > 100 else 0,
        observation_data=observation_data,
        verbose=True,
        seed=0,
    )

    with open(f'samples_states_{N}_particles_{M}_pg_iterations.pickle', 'wb') as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'samples_parameters_{N}_particles_{M}_pg_iterations.pickle', 'wb') as handle:
        pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'samples_loglikelihoods_{N}_particles_{M}_pg_iterations.pickle', 'wb') as handle:
        pickle.dump(loglikelihoods, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # %%
    plt.hist(loglikelihoods, bins=50, density=True)
    plt.xlabel("Log-Likelihood")
    plt.ylabel("Density")
    plt.savefig(f"./figures/sigma_beta_loglikelihoods_{N}_particles_{M}_pg_iterations.pdf", bbox_inches="tight")
    plt.close()


    # %%
    plt.hist(parameters[:, 0], bins=50, density=True, label="Estimated posterior")
    plt.plot([0.16, 0.16], [0, plt.gca().get_ylim()[1]], 'r', label='True value')
    plt.xlabel("sigma")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"./figures/marginal_posterior_sigma_{N}_particles_{M}_pg_iterations.pdf", bbox_inches="tight")
    plt.close()


    # %%
    plt.hist(parameters[:, 1], bins=50, density=True, label="Estimated posterior")
    plt.plot([0.70, 0.70], [0, plt.gca().get_ylim()[1]], 'r', label='True value')
    plt.xlabel("beta")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"./figures/marginal_posterior_beta_{N}_particles_{M}_pg_iterations.pdf", bbox_inches="tight")
    plt.close()
