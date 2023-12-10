import numpy as np
from cmdstanpy import CmdStanMCMC, CmdStanModel
from jaxtyping import Float

from ..waic import calculate_waic


def run_mcmc(
    model: CmdStanModel,
    X: Float[np.ndarray, "N D"],
    y: Float[np.ndarray, " N"],
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    show_progress: bool = False,
    show_console: bool = False,
):
    return model.sample(
        data={
            "N": X.shape[0],
            "D": X.shape[1],
            "X": X,
            "y": y,
            "N_new": 0,
            "X_new": [],
        },
        iter_sampling=num_samples,
        iter_warmup=num_warmup,
        chains=num_chains,
        seed=seed,
        show_progress=show_progress,
        show_console=show_console,
    )


def evaluate_model(
    posterior: CmdStanMCMC,
    logp_var_name: str = "logp",
):
    logp = posterior.draws_xr(logp_var_name)[logp_var_name].to_numpy()
    c, s, n = logp.shape  # chains, samples, n_data
    return calculate_waic(logp.reshape(c * s, n))


def make_prediction(
    model: CmdStanModel,
    X_new: Float[np.ndarray, "N_new D"],
    posterior: CmdStanMCMC,
    y_var_name: str = "y_new",
    seed: int = 0,
    show_console: bool = False,
) -> Float[np.ndarray, "S N_new"]:
    N_new, D = X_new.shape
    gq = model.generate_quantities(
        data={
            "N": 0,
            "D": D,
            "X": [],
            "y": [],
            "N_new": N_new,
            "X_new": X_new,
        },
        previous_fit=posterior,
        seed=seed,
        show_console=show_console,
    )
    y_new = gq.draws_xr(y_var_name)[y_var_name].to_numpy()
    c, s, n = y_new.shape  # chains, samples, n_data
    return y_new.reshape(c * s, n)
