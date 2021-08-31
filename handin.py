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
# # Problem 1: Importance sampling theory [5p]

# %%
def importance_sample(lambd, n_samples: list):
    proposal = scipy.stats.norm(0, 1/lambd)
    target = scipy.stats.norm(0, 1)
    all_weights = []
    all_samples = []
    for N in n_samples:
        samples = proposal.rvs(size=N)
        weights = np.exp(target.logpdf(samples) - proposal.logpdf(samples))
        all_samples.append(samples)
        all_weights.append(weights)
    return all_samples, all_weights


# %%
n_samples = list(range(10, 10000, 10))


# %%
lambd = 1.5
all_samples, all_weights = importance_sample(lambd, n_samples)
normalizing_constants = [np.mean(weights) for weights in all_weights]
plt.plot(n_samples, normalizing_constants)


# %%
lambd = 2.1
all_samples, all_weights = importance_sample(lambd, n_samples)
normalizing_constants = [np.mean(weights) for weights in all_weights]
plt.plot(n_samples, normalizing_constants)


# %%
# variances = []
# for lambd in tqdm([0.1, 1, 1.9, 2.1, 3]):
#     normalizing_constants = []
#     for r in tqdm(range(10)):
#         all_samples, all_weights = importance_sample(lambd, n_samples)
#         normalizing_constants.append([np.mean(weights) for weights in all_weights])  # [[mean(weights) for N in n_samples]]
    
#     normalizing_constants = np.array(normalizing_constants)
#     variances.append(np.var(normalizing_constants, axis=0))

# variances = np.array(variances)
# variances.shape

# plt.plot(n_samples, variances.T)
# plt.yscale("log")


# %%
x = np.linspace(-5, 5, 1000)
n1 = scipy.stats.norm(0, 1).logpdf(x)
plt.plot(x, n1 - scipy.stats.norm(0, 0.3).logpdf(x), label="$\sigma=0.3$")
plt.plot(x, n1 - scipy.stats.norm(0, 0.4).logpdf(x), label="$\sigma=0.4$")
plt.plot(x, n1 - scipy.stats.norm(0, 0.5).logpdf(x), label="$\sigma=0.5$")
plt.plot(x, n1 - scipy.stats.norm(0, 0.75).logpdf(x), label="$\sigma=0.75$")
plt.plot(x, n1 - scipy.stats.norm(0, 1).logpdf(x), label="$\sigma=1$")
plt.plot(x, n1 - scipy.stats.norm(0, 1.5).logpdf(x), label="$\sigma=1.5$")
plt.ylabel("$\log w = \log \mathcal{N}(x|0,1) - \log\mathcal{N}(0,\sigma^2)$")
plt.xlabel("$x$")
plt.legend()
plt.savefig("./figures/importance_sampling_sigma.pdf", bbox_inches="tight")

# %% [markdown]
# # Problem 2: Particle filter for a linear Gaussian state-space model [16p]

# %%
# Constants
A = 0.9  # state transition matrix
Q = 0.5  # state variance
C = 1.3  # observation matrix
R = 0.1  # observation variance

# %% [markdown]
# ## a) Simulate the model
# 

# %%
def step_x(x):
    return A * x + scipy.stats.norm(0, np.sqrt(Q)).rvs()

def step_y(x):
    return C * x + scipy.stats.norm(0, np.sqrt(R)).rvs(x.shape[0])

def simulate(initial_x, step_x_fcn, step_y_fcn, n_timesteps):
    xs = [initial_x] + [None] * n_timesteps
    for t in range(n_timesteps):
        xs[t+1] = step_x_fcn(xs[t])
    
    xs = np.array(xs[1:])
    ys = step_y_fcn(xs)
    return xs, ys


# %%
np.random.seed(0)
T = 2000
initial_x = np.random.normal(0, 1)
x_data, y_data = simulate(initial_x, step_x, step_y, 2000)
x_data.shape


# %%
plt.plot(y_data, label="$y_t$")
plt.plot(x_data, label="$x_t$")
plt.xlabel("T")
plt.legend()


# %%
plt.plot(y_data[100:200], label="$y_t$")
plt.plot(x_data[100:200], label="$x_t$")
plt.xlabel("T")
plt.legend()

# %% [markdown]
# ## b) Kalman Filtering

# %%
def weighted_mean_and_var(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, variance)


# %%
# Constants
P0 = 1  # initial state variance


# %%
np.random.seed(0)
initial_x = 0  # scipy.stats.norm(0, P0).rvs()
initial_Pt_filtering = P0


# %%
def kalman_filter(initial_x, initial_Pt_filtering, y_data, A, C, Q, R):
    T = len(y_data)
    xs = [None] * T + [initial_x]
    Pts_filtering = [None] * T + [initial_Pt_filtering]
    for t in range(T):
        Pt_predictive = A * Pts_filtering[t-1] * A + Q

        Kt = Pt_predictive * C / (C * Pt_predictive * C + R)

        # state update
        xs[t] = A * xs[t-1] + Kt * (y_data[t] - C * A * xs[t-1])

        # variance update
        Pts_filtering[t] = Pt_predictive - Kt * C * Pt_predictive

    xs = np.array(xs[:-1])
    Pts_filtering = np.array(Pts_filtering[:-1])

    output = SimpleNamespace(
        particles=xs,
        variances=Pts_filtering
    )
    return output


# %%
kalman_out = kalman_filter(initial_x, initial_Pt_filtering, y_data, A, C, Q, R)


# %%
kalman_out.particles.shape


# %%
plt.plot(range(200,250,1), kalman_out.particles[200:250], label="Kalman Filter $\hat{x}_t$")
plt.plot(range(200,250,1), x_data[200:250], label="Data $x_t$")
plt.xlabel("$t$")
plt.ylabel("$x_t$")
plt.legend()
plt.savefig("./figures/kalman_filter_particles_zoom.pdf", bbox_inches="tight")


# %%
plt.plot(kalman_out.particles, label="Kalman Filter $\hat{x}_t$")
plt.plot(x_data, label="Data $x_t$")
plt.xlabel("$t$")
plt.ylabel("$x_t$")
plt.legend()
plt.savefig("./figures/kalman_filter_particles.pdf", bbox_inches="tight")


# %%
plt.plot((kalman_out.particles - x_data), label="Mean absolute error of Kalman Filter")
plt.xlabel("$t$")
plt.ylabel("$|\hat{x}_t-x_t|$")
plt.legend()
plt.savefig("./figures/kalman_filter_mean_absolute_error.pdf", bbox_inches="tight")


# %%
plt.plot(kalman_out.variances)

# %% [markdown]
# ## c) Bootstrap Particle Filtering

# %%
def bootstrap_pf(initial_particles, y_data, A, C, Q, R, verbose=True, seed=0):
    """Bootstrap Particle Filter"""
    np.random.seed(seed)
    N = len(initial_particles)
    T = len(y_data)
    if verbose:
        print(f"Running with {N} particles")
    weights = [None] * T + [np.array([1/N] * N)]
    particles = [None] * T + [initial_particles]
    mean_filtering = [None] * T
    var_filtering = [None] * T
    ancestor_indices = [None] * T

    iterator = tqdm(range(T)) if verbose else range(T)
    for t in iterator:
        # RESAMPLE
        ancestor_indices[t] = np.random.choice(range(N), p=weights[t-1], replace=True, size=N)

        # PROPAGATE
        # state
        fcn = A * particles[t-1][ancestor_indices[t]]
        proposal_dist = scipy.stats.norm(fcn, np.sqrt(Q))
        particles[t] = proposal_dist.rvs()
        # measurement
        fcn = C * particles[t]
        measurement_dist = scipy.stats.norm(fcn, np.sqrt(R))

        # WEIGHT
        log_weights_unnorm = measurement_dist.logpdf(y_data[t])
        weights_unnorm = np.exp(log_weights_unnorm - np.max(log_weights_unnorm))
        weights[t] = weights_unnorm / np.sum(weights_unnorm)

        mean_filtering[t], var_filtering[t] = weighted_mean_and_var(particles[t], weights[t])

    weights = np.array(weights[:-1])
    particles = np.array(particles[:-1])
    mean_filtering = np.array(mean_filtering)
    var_filtering = np.array(var_filtering)
    ancestor_indices = np.array(ancestor_indices)
    
    output = SimpleNamespace(
        weights=weights,
        particles=particles,
        mean_filtering=mean_filtering,
        var_filtering=var_filtering,
        ancestor_indices=ancestor_indices,
    )
    
    return output


# %%
N = 500
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))
initial_particles = initial_particle_dist.rvs(N)

bpf_out = bootstrap_pf(initial_particles, y_data, A, C, Q, R)

bpf_out.mean_filtering.shape, bpf_out.ancestor_indices.shape, kalman_out.particles.shape, y_data.shape


# %%
plt.plot(bpf_out.mean_filtering, label="$\hat{x}_t$")
plt.plot(x_data, label="$x_t$")
plt.xlabel("T")
plt.legend()


# %%
plt.plot(bpf_out.mean_filtering[200:250], label="$\hat{x}_t$")
plt.plot(x_data[200:250], label="$x_t$")
plt.xlabel("T")
plt.legend()


# %%
plt.plot((bpf_out.mean_filtering - x_data), label="$|\hat{x}_t - x_t$|")
plt.xlabel("T")
plt.legend()


# %%
plt.plot(bpf_out.var_filtering, label="Var")
plt.legend()

# %% [markdown]
# ### Comparison to the Kalman Filter

# %%
plt.plot(bpf_out.mean_filtering, label="BPF")
plt.plot(kalman_out.particles, label="Kalman")
plt.xlabel("$t$")
plt.ylabel("$\mathbb{E}[p(x_t|y_{1:t})]$")
plt.legend()
plt.savefig("./figures/bpf_kalman_filter_means.pdf", bbox_inches="tight")


# %%
plt.plot(bpf_out.var_filtering, label="BPF")
plt.plot(kalman_out.variances, label="Kalman")
plt.xlabel("$t$")
plt.ylabel("$Var[p(x_t|y_{1:t})]$")
plt.legend()
plt.savefig("./figures/bpf_kalman_filter_variances.pdf", bbox_inches="tight")


# %%

plt.plot(np.abs(bpf_out.mean_filtering - kalman_out.particles))
plt.xlabel("$t$")
plt.ylabel("$|\hat{x}_{t,BPF}-\hat{x}_{t,Kalman}|$")
plt.savefig("./figures/bpf_filter_mean_absolute_error_to_kalman_filter.pdf", bbox_inches="tight")
print("Mean Absolute Error: ", np.mean(np.abs(bpf_out.mean_filtering - kalman_out.particles)))


# %%
plt.plot(np.abs(bpf_out.var_filtering - kalman_out.variances))
plt.xlabel("$t$")
plt.ylabel("$|\widehat{\sigma^2}_{t,BPF}-\widehat{\sigma^2}_{t,Kalman}|$")
plt.savefig("./figures/bpf_filter_var_absolute_error_to_kalman_filter.pdf", bbox_inches="tight")
print("Mean Absolute Error: ", np.mean(np.abs(bpf_out.var_filtering - kalman_out.variances)))


# %%
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))

Ns = [10, 50, 100, 2000, 5000]

bpf_all_mean_filtering = []
bpf_all_var_filtering = []

for N in Ns:
    initial_particles = initial_particle_dist.rvs(N)
    out_bpf = bootstrap_pf(initial_particles, y_data, A, C, Q, R, verbose=1)
    bpf_all_mean_filtering.append(out_bpf.mean_filtering)
    bpf_all_var_filtering.append(out_bpf.var_filtering)


# %%
avg_absolute_differences_of_mean = [np.mean(np.abs(kalman_out.particles - np.array(mean_filtering))) for mean_filtering in bpf_all_mean_filtering]
avg_absolute_differences_of_var = [np.mean(np.abs(kalman_out.variances - np.array(var_filtering))) for var_filtering in bpf_all_var_filtering]


# %%
Ns, avg_absolute_differences_of_mean


# %%
Ns, avg_absolute_differences_of_var

# %% [markdown]
# ## d) Fully Adapted Particle Filtering

# %%
def log_op_exp(array, op=np.mean, axis=-1):
    """Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.

    :param array: Tensor to compute LSE over
    :param axis: dimension to perform operation over
    :param op: reductive operation to be applied, e.g. np.sum or np.mean
    :return: LSE
    """
    maximum = np.max(array, axis=axis)
    return np.log(op(np.exp(array - maximum), axis=axis) + 1e-8) + maximum


def systematic_resampling(w, n_strata=None):
    n_strata = len(w) if n_strata is None else n_strata
    u = (np.arange(n_strata) + np.random.rand())/n_strata
    bins = np.cumsum(w)
    return np.digitize(u, bins)


def multinmomial_resampling(w, N=None):
    N = len(w) if N is None else N
    return np.random.choice(range(N), p=w, replace=True, size=N)


def compute_ess(w):
    return 1 / np.sum(w ** 2)


def fully_adapted_pf(initial_particles, y_data, A, C, Q, R, resampling="multinomial", ess_trigger=None, verbose=0, seed=0):
    """Fully Adapted Particle Filter"""
    if seed is not None:
        np.random.seed(seed)

    N = len(initial_particles)
    T = len(y_data)
    if verbose:
        print(f"Running with {N} particles")

    nu_weights = [None] * T  # these are nu weights
    particles = [None] * T + [initial_particles]  # draw initial particles - put at index -1
    mean_observation = [None] * T  # p(y_t|x_t)
    var_observation = [None] * T
    mean_state_prediction = [None] * T  # p(x_t|x_t-1)
    var_state_prediction = [None] * T
    mean_filtering = [None] * T  # p(x_t|x_t-1, y_t)
    var_filtering = [None] * T
    ancestor_indices = [None] * T
    effective_sample_size = [None] * T
    loglikelihood = 0
    
    if ess_trigger is None:
        ess_trigger = N
        N_ess = 0
        do_adaptive_resample = False
    else:
        do_adaptive_resample = True
    
    importance_weights = np.array([1/N] * N)
    
    if resampling == "multinomial":
        resample = multinmomial_resampling
    elif resampling == "systematic":
        resample = systematic_resampling
    else:
        raise ValueError(f"Unknown resampling method: {resampling}")
    
    K = Q * C / (C * Q * C + R)
    state_proposal_stddev = np.sqrt((1 - K * C) * Q)
    obs_proposal_stddev = np.sqrt(C * Q * C + R)
    
    iterator = tqdm(range(T)) if verbose else range(T)
    for t in iterator:
        # WEIGHT
        # measurement
        fcn_weight = A * particles[t-1]
        mean = C * fcn_weight
        measurement_proposal_dist = scipy.stats.norm(mean, obs_proposal_stddev)

        # compute weights (nu)
        log_nu_weights_unnorm = measurement_proposal_dist.logpdf(y_data[t])
        nu_weights_unnorm = np.exp(log_nu_weights_unnorm - np.max(log_nu_weights_unnorm))
        nu_weights[t] = nu_weights_unnorm / np.sum(nu_weights_unnorm)

        # RESAMPLE
        if do_adaptive_resample:
            N_ess = compute_ess(nu_weights[t])
            effective_sample_size[t] = N_ess
        if N_ess < ess_trigger:
            a_indices = resample(nu_weights[t])
        else:
            a_indices = np.arange(N)
        ancestor_indices[t] = a_indices

        # PROPAGATE
        # state
        fcn_prop = fcn_weight[a_indices]
        mean = fcn_prop + K * (y_data[t] - C * fcn_prop)
        state_proposal_dist = scipy.stats.norm(mean, state_proposal_stddev)
        particles[t] = state_proposal_dist.rvs()
        # measurement (optional)
        measurement_dist = scipy.stats.norm(C * np.mean(particles[t]), np.sqrt(R))
        mean_observation[t] = measurement_dist.mean()
        var_observation[t] = measurement_dist.var()

        mean_filtering[t], var_filtering[t] = weighted_mean_and_var(particles[t], importance_weights)

        state_prediction_dist = scipy.stats.norm(fcn_weight, np.sqrt(Q))  # prediction formed by ignoring y_data (not available)
        mean_state_prediction[t] = np.mean(state_prediction_dist.mean())
        var_state_prediction[t] = np.mean(state_prediction_dist.var())

        # likelihood
        log_obs = measurement_dist.logpdf(y_data[t])
        log_state_pred = state_prediction_dist.logpdf(particles[t])
        log_state_prop = state_proposal_dist.logpdf(particles[t])
        loglikelihood_term = log_obs + log_state_pred - log_state_prop - np.log(nu_weights[t][a_indices]) - np.log(N)
        loglikelihood += log_op_exp(loglikelihood_term, np.mean)

    nu_weights = np.array(nu_weights)
    particles = np.array(particles[:-1])
    mean_filtering = np.array(mean_filtering)
    var_filtering = np.array(var_filtering)
    mean_state_prediction = np.array(mean_state_prediction)
    var_state_prediction = np.array(var_state_prediction)
    mean_observation = np.array(mean_observation)
    var_observation = np.array(var_observation)
    loglikelihood = np.array(loglikelihood)
    ancestor_indices = np.array(ancestor_indices)
    effective_sample_size = np.array(effective_sample_size)

    output = SimpleNamespace(
        nu_weights=nu_weights,
        particles=particles,
        mean_filtering=mean_filtering,
        var_filtering=var_filtering,
        mean_state_prediction=mean_state_prediction,
        var_state_prediction=var_state_prediction,
        mean_observation=mean_observation,
        var_observation=var_observation,
        loglikelihood=loglikelihood,
        ancestor_indices=ancestor_indices,
        effective_sample_size=effective_sample_size,
    )
    return output


# %%
N = 500
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))
initial_particles = initial_particle_dist.rvs(N)

fapf_out = fully_adapted_pf(initial_particles, y_data, A, C, Q, R, verbose=1, seed=0)

fapf_out.mean_filtering.shape, fapf_out.ancestor_indices.shape

# %% [markdown]
# ### Comparison to the Bootstrap Particle Filter

# %%
plt.plot(bpf_out.mean_filtering, label="BPF filtering")
plt.plot(fapf_out.mean_filtering, label="FAPF filtering")
plt.plot(kalman_out.particles, label="Kalman filtering")
plt.legend()


# %%
plt.plot(bpf_out.var_filtering, label="BPF variance")
plt.plot(fapf_out.var_filtering, label="FAPF variance")
plt.plot(kalman_out.variances, label="Kalman variance")
plt.legend()


# %%
plt.plot(np.abs(bpf_out.mean_filtering - kalman_out.particles), label="BPF")
plt.plot(np.abs(fapf_out.mean_filtering - kalman_out.particles), label="AFPF")
plt.xlabel("$t$")
plt.ylabel("$|\hat{x}_{t}-\hat{x}_{t,Kalman}|$")
plt.legend()
plt.savefig("./figures/fapf_mean_absolute_error_to_bpf.pdf", bbox_inches="tight")
print("Mean Absolute Error (BPF): ", np.mean(np.abs(bpf_out.mean_filtering - kalman_out.particles)))
print("Mean Absolute Error (FAPF): ", np.mean(np.abs(fapf_out.mean_filtering - kalman_out.particles)))


# %%
plt.plot(np.abs(bpf_out.var_filtering - kalman_out.variances), label="BPF")
plt.plot(np.abs(fapf_out.var_filtering - kalman_out.variances), label="AFPF")
plt.xlabel("$t$")
plt.ylabel("$|\widehat{\sigma^2}_{t,BPF}-\widehat{\sigma^2}_{t,Kalman}|$")
plt.legend()
plt.savefig("./figures/fapf_var_absolute_error_to_bpf.pdf", bbox_inches="tight")
print("Mean Absolute Error (BPF): ", np.mean(np.abs(bpf_out.var_filtering - kalman_out.variances)))
print("Mean Absolute Error (FAPF): ", np.mean(np.abs(fapf_out.var_filtering - kalman_out.variances)))

# %% [markdown]
# ### Comparison to the Kalman Filter

# %%
plt.plot(fapf_out.mean_filtering, label="FAPF")
plt.plot(kalman_out.particles, label="Kalman")
plt.xlabel("$t$")
plt.ylabel("$\mathbb{E}[p(x_t|y_{1:t})]$")
plt.legend()
plt.savefig("./figures/fapf_kalman_filter_means.pdf", bbox_inches="tight")


# %%
plt.plot(fapf_out.var_filtering, label="FAPF")
plt.plot(kalman_out.variances, label="Kalman")
plt.xlabel("$t$")
plt.ylabel("$Var[p(x_t|y_{1:t})]$")
plt.legend()
plt.savefig("./figures/fapf_kalman_filter_variances.pdf", bbox_inches="tight")


# %%

plt.plot(np.abs(fapf_out.mean_filtering - kalman_out.particles))
plt.xlabel("$t$")
plt.ylabel("$|\hat{x}_{t,FAPF}-\hat{x}_{t,Kalman}|$")
plt.savefig("./figures/fapf_filter_mean_absolute_error_to_kalman_filter.pdf", bbox_inches="tight")
print("Mean Absolute Error: ", np.mean(np.abs(fapf_out.mean_filtering - kalman_out.particles)))


# %%
plt.plot(np.abs(fapf_out.var_filtering - kalman_out.variances))
plt.xlabel("$t$")
plt.ylabel("$|\widehat{\sigma^2}_{t,FAPF}-\widehat{\sigma^2}_{t,Kalman}|$")
plt.savefig("./figures/fapf_filter_var_absolute_error_to_kalman_filter.pdf", bbox_inches="tight")
print("Mean Absolute Error: ", np.mean(np.abs(fapf_out.var_filtering - kalman_out.variances)))


# %%
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))

Ns = [10, 50, 100, 2000, 5000]

fapf_all_mean_filtering = []
fapf_all_var_filtering = []

for N in Ns:
    initial_particles = initial_particle_dist.rvs(N)
    fapf_out = fully_adapted_pf(initial_particles, y_data, A, C, Q, R, verbose=1, seed=0)
    fapf_all_mean_filtering.append(fapf_out.mean_filtering)
    fapf_all_var_filtering.append(fapf_out.var_filtering)


# %%
avg_absolute_differences_of_mean = [np.mean(np.abs(kalman_out.particles - np.array(mean_filtering))) for mean_filtering in fapf_all_mean_filtering]
avg_absolute_differences_of_var = [np.mean(np.abs(kalman_out.variances - np.array(var_filtering))) for var_filtering in fapf_all_var_filtering]


# %%
Ns, avg_absolute_differences_of_mean


# %%
Ns, avg_absolute_differences_of_var

# %% [markdown]
# ## e) Genealogy of Fully Adapted Particle Filtering

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
def plot_genealogy(genealogy, particles, ancestor_indices, t1=None, t2=None, reference_trajectory=None, sampled_trajectory=None, verbose=0, figsize=(20, 10), alpha=0.3):
    """Requires initial particle to be at particles[-1] and len(particles) = len(ancestor_indices) + 1"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if t1 is None:
        t1 = 0
    if t2 is None:
        t2 = len(ancestor_indices)  # T == len(y_data)

    assert t2 <= len(ancestor_indices)

    T = t2 - t1
    
    # [t1, t1+1, ..., t2-2, t2-1]
    iterator = tqdm(range(t1, t2)) if verbose else range(t1, t2)
    for i, t in enumerate(iterator):
        p = np.array([particles[t-1][ancestor_indices[t]], particles[t]])
        ax.plot([i, i+1], p, marker='o', color='silver', alpha=alpha)

    # plot first particles (initial) without ancestral resampling (the above makes some particles at t1-1 not appear)
    ax.plot([0, 0], [particles[t1-1,:-1], particles[t1-1,:-1]], marker='o', color='silver', alpha=alpha)
    ax.plot([0, 0], [particles[t1-1,-1:], particles[t1-1,-1:]], marker='o', color='silver', alpha=alpha, label="Particles")
    if verbose:
        print("Plotted particles and connecting ancestral lines")

    # [t1, t1+1, ..., t2-1, t2]
    ax.plot(genealogy[t1:t2+1 ,:-1], marker='o', color='tab:red')
    ax.plot(genealogy[t1:t2+1 ,-1:], marker='o', color='tab:red', label="Genealogy")
    if verbose:
        print("Plotted genealogy")
    
    if reference_trajectory is not None:
        # put initial at front
        reference_trajectory_plot = np.concatenate([reference_trajectory[-1:], reference_trajectory[:-1]])
        ax.plot(reference_trajectory_plot[t1:t2], color="tab:blue", label="Reference trajectory")
        if verbose:
            print("Plotted reference tracjectory")

    if sampled_trajectory is not None:
        # put initial at front
        sampled_trajectory_plot = np.concatenate([sampled_trajectory[-1:], sampled_trajectory[:-1]])
        ax.plot(sampled_trajectory_plot[t1:t2], color="tab:green", label="Sampled trajectory")
        if verbose:
            print("Plotted sampled tracjectory")


    ticks = np.linspace(0, T, 11).astype(int).tolist()
    labels = np.linspace(t1, t2, 11).astype(int).tolist()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.legend()
    return fig, ax


# %%
N = 100
t = T
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))
initial_particles = initial_particle_dist.rvs(N)

fapf_out = fully_adapted_pf(initial_particles, y_data[:t], A, C, Q, R, verbose=1, seed=0)


# %%
# Append initial particle to list of particles
if fapf_out.particles.shape[0] == T:
    fapf_out.particles = np.concatenate([fapf_out.particles, initial_particles[np.newaxis]], axis=0)


# %%
fapf_out.particles.shape, fapf_out.ancestor_indices.shape


# %%
fapf_out.genealogy = backtrack_genealogy(fapf_out.ancestor_indices, fapf_out.particles)
genealogy = fapf_out.genealogy


# %%
genealogy.shape, fapf_out.particles.shape, fapf_out.ancestor_indices.shape


# %%
t1 = T-10
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_out.particles, fapf_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = T-30
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_out.particles, fapf_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = T-100
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_out.particles, fapf_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = 0
t2 = 30
fig, ax = plot_genealogy(genealogy, fapf_out.particles, fapf_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')

# %% [markdown]
# ## f) Genealogy of Fully Adapted Particle Filtering with Systematic Resampling

# %%
N = 100
t = T
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))
initial_particles = initial_particle_dist.rvs(N)

fapf_sys_out = fully_adapted_pf(initial_particles, y_data[:t], A, C, Q, R, resampling="systematic", verbose=1, seed=0)

# Append initial particle to list of particles
fapf_sys_out.particles = np.concatenate([fapf_sys_out.particles, initial_particles[np.newaxis]], axis=0)

genealogy = backtrack_genealogy(fapf_sys_out.ancestor_indices, fapf_sys_out.particles)

fapf_sys_out.genealogy = genealogy

genealogy.shape, fapf_sys_out.particles.shape, fapf_sys_out.ancestor_indices.shape


# %%
t1 = T-10
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_sys_out.particles, fapf_sys_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_sys_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = T-100
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_sys_out.particles, fapf_sys_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_sys_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = 0
t2 = 30
fig, ax = plot_genealogy(genealogy, fapf_sys_out.particles, fapf_sys_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_sys_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')

# %% [markdown]
# ## g) Genealogy of Fully Adapted Particle Filtering with Systematic and Adaptive Resampling

# %%
N = 100
t = T
initial_particle_dist = scipy.stats.norm(0, np.sqrt(P0))
initial_particles = initial_particle_dist.rvs(N)

fapf_sys_adap_out = fully_adapted_pf(initial_particles, y_data[:t], A, C, Q, R, resampling="systematic", ess_trigger=N//2, verbose=1, seed=0)

# Append initial particle to list of particles
fapf_sys_adap_out.particles = np.concatenate([fapf_sys_adap_out.particles, initial_particles[np.newaxis]], axis=0)

genealogy = backtrack_genealogy(fapf_sys_adap_out.ancestor_indices, fapf_sys_adap_out.particles)

fapf_sys_adap_out.genealogy = genealogy

genealogy.shape, fapf_sys_adap_out.particles.shape, fapf_sys_adap_out.ancestor_indices.shape


# %%
fapf_sys_adap_out.effective_sample_size


# %%
t1 = 0
t2 = 30
fig, ax = plot_genealogy(genealogy, fapf_sys_adap_out.particles, fapf_sys_adap_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_sys_adap_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = T-10
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_sys_adap_out.particles, fapf_sys_adap_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_sys_adap_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
t1 = T-100
t2 = T
fig, ax = plot_genealogy(genealogy, fapf_sys_adap_out.particles, fapf_sys_adap_out.ancestor_indices, t1=t1, t2=t2, verbose=True)
plt.xlabel("$t$")
plt.ylabel("$x_t$")
fig.savefig(f"./figures/afpf_sys_adap_genealogy_{N}_particles_timesteps_{t1}_to_{t2}.pdf", bbox_inches='tight')


# %%
plt.plot(fapf_sys_adap_out.effective_sample_size)


# %%
plt.plot(fapf_sys_adap_out.effective_sample_size / N, label="$N_{eff}\,/\,N$")
plt.plot([0, T], [0.5, 0.5], label="Threshold")
plt.xlabel("$t$")
plt.ylabel("$N_{eff}\,/\,N$")
plt.legend()
plt.savefig(f"./figures/afpf_sys_adap_Neff_ratio_{N}_particles.pdf", bbox_inches='tight')

# %% [markdown]
# ### Compare the number of unique paths in genealogy

# %%
def compute_n_paths(genealogy, axis=1):
    b = np.sort(genealogy, axis=axis)
    n_paths = (b[:, 1:] != b[:, :-1]).sum(axis=axis) + 1
    return n_paths


# %%
plt.plot(compute_n_paths(fapf_out.genealogy), label="Multinomial resamplnig")
plt.plot(compute_n_paths(fapf_sys_out.genealogy), label="Systematic resampling")
plt.plot(compute_n_paths(fapf_sys_adap_out.genealogy), label="Systematic resampling with adaptive resampling")
plt.xlabel("Number of paths in genealogy")
plt.ylabel("x_t")
plt.legend()
plt.yscale("log")
plt.savefig(f"./figures/afpf_sys_adap_number_of_active_paths_{N}_particles.pdf", bbox_inches='tight')

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


# %%
params[np.argmax(loglikelihoods)], np.argmax(loglikelihoods)


# %%
plt.hist(params[:,0], bins=50, density=True, label="Estimated posterior")
plt.plot([0.16, 0.16], [0, plt.gca().get_ylim()[1]], 'r', label='True value')
plt.xlabel("sigma")
plt.ylabel("Density")
plt.legend()
plt.savefig(f"./figures/marginal_posterior_sigma_{N}_particles_{M}_pmh_iterations.pdf", bbox_inches="tight")


# %%
plt.hist(params[:,1], bins=50, density=True, label="Estimated posterior")
plt.plot([0.70, 0.70], [0, plt.gca().get_ylim()[1]], 'r', label='True value')
plt.xlabel("beta")
plt.ylabel("Density")
plt.legend()
plt.savefig(f"./figures/marginal_posterior_beta_{N}_particles_{M}_pmh_iterations.pdf", bbox_inches="tight")


# %%
plt.scatter(params[:,0], loglikelihoods);


# %%
plt.scatter(params[:,1], loglikelihoods);

# %% [markdown]
# ## c) Particle Gibbs for parameter estimation
# %% [markdown]
# # Problem 4: SMC sampler [8p]

# %%



# %%



# %%



