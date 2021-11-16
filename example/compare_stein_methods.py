# %        print(ytr_std)%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import numpyro
from numpyro.contrib.callbacks import Progbar
from numpyro.contrib.einstein import Stein, RBFKernel, LinearKernel
from numpyro.distributions import Normal, NormalMixture
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal

plt.rcParams.update({"font.size": 22, "axes.linewidth": 3})
TARGET_1D = NormalMixture(jnp.array([1 / 3, 2 / 3]), jnp.array([-2.0, 2.0]), jnp.array([1.0, 1.0]))


def model():
    numpyro.sample("x", TARGET_1D)  # 1/3 N(-2,1) + 2/3 N(2,1)


def main():
    rng_key = jax.random.PRNGKey(42)
    num_iterations = 3000

    # SVGD
    svgd = Stein(model, AutoDelta(model), numpyro.optim.Adagrad(step_size=1.0), Trace_ELBO(), RBFKernel(),
                 num_particles=100)

    state, _ = svgd.run(rng_key, num_iterations, callbacks=[Progbar()])
    plot(loc=svgd.get_params(state)["x_auto_loc"], name='svgd')

    # SVI
    svi = SVI(model, AutoNormal(model), numpyro.optim.Adagrad(step_size=1.0), Trace_ELBO())
    params, _, _ = svi.run(rng_key, num_iterations)
    plot(loc=params["x_auto_loc"], scale=params["x_auto_scale"], name='svi', particle_method=False)

    for num_particles, kernel, lr in zip((2, 3), (LinearKernel(), RBFKernel()), (.1, 1.)):  # Stein mixture
        svgd = Stein(model, AutoNormal(model), numpyro.optim.Adagrad(step_size=lr), Trace_ELBO(), kernel,
                     num_particles=num_particles)
        state, _ = svgd.run(rng_key, num_iterations, callbacks=[Progbar()])
        plot(loc=svgd.get_params(state)["x_auto_loc"], scale=svgd.get_params(state)["x_auto_scale"],
             name=f'stein_mixture_p{num_particles}')


def plot(loc, scale=None, name='', particle_method=True):
    fig, ax = plt.subplots()
    x = jnp.linspace(-8, 8.0, 200)
    ax.plot(
        x,
        jnp.exp(TARGET_1D.log_prob(x)),
        label="target_dist",
        color="b",
        linewidth=7,
        linestyle="dashed",
    )

    ax.scatter(loc, np.zeros_like(loc), color="g", zorder=9, s=100)
    if particle_method:
        if scale is None:
            sns.kdeplot(x=loc, color="g", ax=ax, linewidth=7)
        else:

            ax.plot(x, jnp.exp(NormalMixture(jnp.ones(scale.shape) / scale.shape[0], loc, scale, ).log_prob(x)),
                    label="target_dist", color="g", linewidth=7)

    else:
        ax.plot(x, jnp.exp(Normal(loc, scale).log_prob(x)), label="target_dist",
                color="g", linewidth=7, zorder=10)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    plt.ylabel("")
    plt.ylim([-0.01, 0.3])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(f"{name}.png")


if __name__ == "__main__":
    main()
