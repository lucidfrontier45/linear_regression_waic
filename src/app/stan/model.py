from pathlib import Path

import cmdstanpy


def get_model():
    dir_path = Path(__file__).parent
    stan_file = dir_path.joinpath("model.stan")
    return cmdstanpy.CmdStanModel(
        stan_file=str(stan_file),
        stanc_options={"O1": True},
        cpp_options={"O3": True},
    )
