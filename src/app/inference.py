import jax
import numpy as np
from jaxtyping import Array, Float, Float32
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood
from scipy.special import logsumexp


def run_mcmc(
    model,
    X: Float32[Array, "N D"],
    y: Float32[Array, " N"],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, X, y)
    return mcmc


def calc_waic(logp: Float[Array, "M D"]) -> float:
    M = logp.shape[0]  # number of samples
    T = -logsumexp(logp, axis=0, b=1.0 / M).mean()
    V = logp.var(axis=0).mean()
    return T + V


def evaluate_model(
    model,
    X: Float32[Array, "N D"],
    y: Float32[Array, " N"],
    posterior_samples: dict[str, Float32[Array, "M _*"]],
):
    logp = log_likelihood(model, posterior_samples, X, y)["y"]
    return calc_waic(jax.device_get(logp))
