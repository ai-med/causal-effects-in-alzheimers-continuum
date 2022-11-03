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
import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from ..pystan_models.models import PPCAExtended, BPMFExtended
from .data import get_volume_causes
from .io import SubstituteConfounderData, load_patient_data, write_deconfounder_data

LOG = logging.getLogger(__name__)


class DeconfounderEstimator:
    def __init__(self, method: str, *, latent_dim: int, random_state: int, silent: bool = False) -> None:
        self.method = method
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.silent = silent

    def get_deconfounder(self, data_volumes: pd.DataFrame, data_extra: pd.DataFrame) -> Tuple[SubstituteConfounderData, Any]:
        assert len(data_volumes.index.symmetric_difference(data_extra.index)) == 0

        if len(data_volumes.columns.intersection(data_extra.columns)) != 0:
            raise ValueError("column names must be unique")

        vol_causes = data_extra.columns.tolist()
        data = pd.concat((data_extra, data_volumes), axis=1)

        if self.method == "ppca":
            cls = PPCAExtended
            subst_conf_name = "Z_mu"
        elif self.method == "bpmf":
            cls = BPMFExtended
            subst_conf_name = "U"
        else:
            raise ValueError(f"{self.method} is not supported")

        m = cls(known_causes=vol_causes, latent_dim=self.latent_dim, random_state=self.random_state)
        m.fit(data)

        pval = m.check_model(silent=self.silent)
        LOG.info("Overall p-value: %f", pval)

        posterior = m.posterior_mean_estimates()

        # approximate the (random variable) substitute confounders with their inferred mean.
        subst_conf = getattr(posterior, subst_conf_name)
        Z_hat = pd.DataFrame(subst_conf, index=data.index).add_prefix("Deconf_")

        recon = m.mean_reconstruction(subst_conf)
        resid = data_volumes.values - recon
        residuals = pd.DataFrame(resid, index=data.index, columns=data_volumes.columns).add_suffix("-resid")

        metadata = pd.DataFrame.from_dict(
            {
                "pvalue": [pval],
                "latent_dim": [self.latent_dim],
                "random_state": [self.random_state],
            }
        )
        metadata.index = [self.method]

        output = SubstituteConfounderData(
            substitute_confounder=Z_hat,
            residuals=residuals,
            metadata=metadata,
        )

        return output, posterior


def load_data(filename):
    data = load_patient_data(filename)

    measurements = pd.concat((
        data.volumes,
        data.thickness
    ), axis=1)
    confounders = data.demographics
    return measurements, confounders


def make_deconfounder(args):
    out_file = args.output_file
    if out_file.exists():
        raise RuntimeError(f"{out_file} already exists.")

    output_dir = out_file.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    deconf = DeconfounderEstimator(args.model, latent_dim=args.latent_dim, random_state=args.random_state, silent=args.silent)

    vols, confounders = load_data(args.input_data)

    vol_causes = get_volume_causes(confounders)
    outputs, posterior = deconf.get_deconfounder(vols, vol_causes)

    write_deconfounder_data(outputs, posterior, out_file)


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    deconf_parser = subparsers.add_parser("deconf")
    deconf_parser.add_argument("-i", "--input_data", type=Path, required=True)
    deconf_parser.add_argument("-m", "--model", choices=["bpmf", "ppca"], required=True)
    deconf_parser.add_argument("-o", "--output_file", type=Path, required=True)
    deconf_parser.add_argument("--latent_dim", type=int, default=2)
    deconf_parser.add_argument("--random_state", type=int, default=2501)
    deconf_parser.add_argument("--silent", action="store_true", default=False)
    deconf_parser.set_defaults(func=make_deconfounder)

    args = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)
    np.seterr("raise")

    if not hasattr(args, "func"):
        parser.error("argument missing")

    args.func(args)


if __name__ == "__main__":
    main()
