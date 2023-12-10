import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float32


def linear_model(X: Float32[Array, "N D"], y: Float32[Array, " N"] | None):
    N, D = X.shape

    with numpyro.plate("dimension", D):
        w = numpyro.sample("w", dist.Normal(0, 1))

    sigma = numpyro.sample("sigma", dist.HalfCauchy(1))

    z = jnp.dot(X, w)  # type: ignore

    with numpyro.plate("data", N):
        y = numpyro.sample("y", dist.Normal(z, sigma), obs=y)  # type: ignore
