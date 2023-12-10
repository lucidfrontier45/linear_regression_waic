import numpy as np
from jaxtyping import Float
from scipy.special import logsumexp


def calculate_waic(logp: Float[np.ndarray, "M D"]) -> float:
    M = logp.shape[0]  # number of samples
    T = -logsumexp(logp, axis=0, b=1.0 / M).mean()  # type: ignore
    V = logp.var(axis=0).mean()
    return T + V
