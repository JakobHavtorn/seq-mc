import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def visualize_sampling_1d(samples, target_dist):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 9))
    x_vals = torch.linspace(samples.min(), samples.max(), 100)
    true_pdf = target_dist.log_prob(x_vals).exp()
    ax1.plot(x_vals, true_pdf, label="Target distribution")
    sns.distplot(ax=ax1, a=samples, label="Approximated distribution")
    ax1.legend()
    return fig, ax1


def visualize_sampling_2d(samples, target_dist, connect_samples=None):
    """Visualize a sampling in 2D.

    Args:
        samples (torch.Tensor): Of shape [samples, 2]
        target_dist (torch.distributions.Distribution): Target distribution to sample from
        connect_samples (torch.Tensor): None or a torch.Tensor of shape [samples, 2] or [steps, samples, 2]
    """
    n_axes = 2 + int(connect_samples is not None)
    fig, axes = plt.subplots(1, n_axes, figsize=(16, 9))

    true_samples = target_dist.sample(sample_shape=(1000,)).numpy()

    # Create third plot of connected samples
    if connect_samples is not None:
        sns.kdeplot(ax=axes[2], x=true_samples[:, 0], y=true_samples[:, 1], label="Target distribution")
        if connect_samples.ndim == 3:  # A set of steps per sample in two dimensions i.e. [steps, samples, 2]
            cm = plt.get_cmap("inferno")
            color_indices = np.ceil(np.linspace(0, len(cm.colors) - 1, len(connect_samples))).astype(np.int32)
            for i, (samples_path, c_index) in enumerate(zip(connect_samples, color_indices)):
                # Append end point of previous leapfrog chain to connect them, if not at the first step.
                if c_index != 0:
                    samples_path = np.concatenate([connect_samples[i - 1, -1:], samples_path], axis=0)
                axes[2].plot(samples_path[:, 0], samples_path[:, 1], color=cm.colors[c_index])
        elif connect_samples.ndim == 2:
            axes[2].plot(connect_samples[:, 0], connect_samples[:, 1], label="Paths")
        else:
            raise ValueError("Can" "t work with samples of shape {}. Expected 2 or 3 dimensions.".format(samples.shape))
        axes[2].legend()

    sns.kdeplot(ax=axes[0], x=true_samples[:, 0], y=true_samples[:, 1], label="Target distribution")
    sns.kdeplot(ax=axes[0], x=samples[:, 0], y=samples[:, 1], label="Approximated distribution")

    sns.kdeplot(ax=axes[1], x=true_samples[:, 0], y=true_samples[:, 1], label="Target distribution")
    axes[1].scatter(x=samples[:, 0], y=samples[:, 1], s=10, c=list(range(samples.shape[0])), label="Samples")

    axes[0].legend()
    axes[1].legend()
    return fig, axes


def plot_autocorrelations(names, autocorrelations, T_cutoff=None):
    """Plot the autocorrelations indicating the zero line and high-variance cutoff.

    Args:
        names (list): Names of the estimands in the autocorrelations (used for labels).
        autocorrelations (torch.Tensor or np.ndarray): Autocorrelations of shape [n_samples, n_estimands].
        T_cutoff (int, optional): The temporal index where autocorrelations first turn negative. Defaults to None.

    Returns:
        tuple: Figure and axis of the plot
    """
    fig, ax = plt.subplots(1, 1)
    lines = ax.plot(autocorrelations)
    if T_cutoff is not None:
        colors = [l.get_color() for l in lines]
        ax.vlines(x=T_cutoff, ymin=-1, ymax=ax.get_ylim()[1], colors=colors, linestyles="dashed")
    ax.axhline(y=0, color="k")
    ax.set_ylim([-1, ax.get_ylim()[1]])
    ax.set_xlabel("$t$")
    ax.set_ylabel("Autocorrelation, $\\rho_t$")
    ax.legend(names + ["$\\rho_t=0$", "$t$ where $\\rho_t<0$"])
    return fig, ax


def contour_plot_2d(samples):
    xmin, ymin = samples.min(0).values
    xmax, ymax = samples.max(0).values

    delta = 0.025
    x = np.arange(xmin * 0.9, xmax * 1.1, delta)
    y = np.arange(ymin * 0.9, ymax * 1.1, delta)
    X, Y = np.meshgrid(x, y)

    Z = np.empty(X.shape + (2,))
    Z[:, :, 0] = X
    Z[:, :, 1] = Y

    import matplotlib

    matplotlib.rcParams["contour.negative_linestyle"] = "solid"
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(X, Y, Z, colors="b")
