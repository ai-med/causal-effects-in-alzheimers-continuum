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

import numpy as np

from ..ukb.substitute_confounder import DeconfounderEstimator
from ..ukb.io import write_deconfounder_data
from .data import get_volume_causes
from .io import load_adni_data

LOG = logging.getLogger(__name__)


def make_deconfounder(args):
    if args.output_file.exists():
        raise RuntimeError(f"{args.output_file} already exists.")

    output_dir = args.output_file.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    deconf = DeconfounderEstimator(args.model, latent_dim=args.latent_dim, random_state=args.random_state, silent=args.silent)

    measures, clinical, _ = load_adni_data(args.input_data)
    causes = get_volume_causes(clinical)

    outputs, posterior = deconf.get_deconfounder(measures, causes)

    write_deconfounder_data(outputs, posterior, args.output_file)


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
