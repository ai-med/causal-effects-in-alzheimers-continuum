# This file is part of Estimation of Causal Effects in the Alzheimer's Continuum (Causal-AD).
#
# Causal-AD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Causal-AD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Causal-AD. If not, see <https://www.gnu.org/licenses/>.
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import papermill as pm

from .configs import get_var_config

LOG = logging.getLogger(__name__)


def get_experiment_parameters(args):
    configs = get_var_config(noise_var=args.noise_variance, n_repeats=args.n_repeats)
    location_tmpl = "simulate_data_x{var_x:d}_z{var_z:d}_eps{var_eps:.4f}_target"
    output_base_dir = Path(args.output_dir)

    for config in configs:
        output_dir = output_base_dir / location_tmpl.format(
            var_x=config.ratio_x_z[0],
            var_z=config.ratio_x_z[1],
            var_eps=config.var_eps,
        ) / f"{config.seed:010d}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        output_file = output_dir / "data_generated.h5"

        params = {
            "data_path": str(Path(args.data_path).resolve()),
            "var_x": config.var_x,
            "var_z": config.var_z,
            "random_state": config.seed,
            "output_file": str(output_file.resolve()),
        }

        yield params


def simulate_data_with_parameters(parameters: Dict[str, Any]) -> None:
    LOG.info("Executing notebook with parameters:\n%s", parameters)
    output_dir = Path(parameters["output_file"]).parent
    output_file = output_dir / "ukb_02_simulate_data_output.ipynb"

    pm.execute_notebook(
        Path(__file__).parent / "notebooks" / "ukb_02_simulate_data.ipynb",
        output_file,
        parameters=parameters,
    )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=Path, required=True, help="Local path to HDF5 file with volume and thickness measurements."
    )
    parser.add_argument(
        "--n_repeats", type=int, required=True, help="Number of repetitions per configuration."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Path to directory where to store simulations."
    )
    parser.add_argument(
        "--noise_variance", type=float, default=0.2, help="Amount of variance attributed to noise. Default: %(default)s"
    )

    args = parser.parse_args(args=args)

    for params in get_experiment_parameters(args):
        simulate_data_with_parameters(params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
