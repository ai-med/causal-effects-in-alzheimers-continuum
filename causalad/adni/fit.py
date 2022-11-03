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
from pathlib import Path
import logging

import pandas as pd
import papermill as pm
import patsy

from .combat import fit_combat
from .data import get_site_id_from_index, get_volume_causes
from .io import load_adni_data
from ..ukb.estimate import fit_regress_out
from ..ukb.io import load_deconfounder_residuals

LOG = logging.getLogger(__name__)


def get_substitute_confounder_inputs(features_dir: Path):
    if not features_dir.exists():
        raise ValueError(f"{features_dir} does not exist")
    for path in features_dir.glob("*.h5"):
        if "ppca" in path.stem or "bpmf" in path.stem:
            yield path


def create_substitute_confounder_inputs(features_dir: str, outcome_file: str, outputs_base_dir: str):
    """"Load residuals and ADAS13 from HDF5 file and write them as CSV"""
    features_dir = Path(features_dir)
    output_dir = Path(outputs_base_dir)
    _, _, outcome = load_adni_data(outcome_file)

    for hdf5_path in get_substitute_confounder_inputs(features_dir):
        residuals = load_deconfounder_residuals(hdf5_path)

        data = pd.concat((residuals, outcome), axis=1)
        out_file = output_dir / f"{hdf5_path.stem}.csv"
        data.to_csv(out_file)

        yield str(out_file.resolve())


def create_original_inputs(outcome_file: str, outputs_base_dir: str) -> str:
    """Original volume and thickness measures without any confounding correction"""
    measurements, _, outcome = load_adni_data(outcome_file)
    output_dir = Path(outputs_base_dir)

    data = pd.concat((measurements, outcome), axis=1)
    out_file = output_dir / "adni_original.csv"
    data.to_csv(out_file)

    return str(out_file.resolve())


def create_age_residualized_inputs(outcome_file: str, outputs_base_dir: str) -> str:
    """Volume and thickness residualized for observed confounders"""
    measurements, clinical, outcome = load_adni_data(outcome_file)
    output_dir = Path(outputs_base_dir)

    df = patsy.dmatrix("AGE + I(AGE**2) - 1", data=clinical, return_type="dataframe")
    df = (df - df.mean()) / df.std(ddof=1)

    residuals = fit_regress_out(measurements, df).add_suffix("-resid")

    data = pd.concat((residuals, outcome), axis=1)
    out_file = output_dir / "adni_age_residualized.csv"
    data.to_csv(out_file)

    return str(out_file.resolve())


def create_combat_residualized_inputs(outcome_file: str, outputs_base_dir: str) -> str:
    """Volume and thickness harmonized with ComBat"""
    measurements, clinical, outcome = load_adni_data(outcome_file)
    output_dir = Path(outputs_base_dir)

    site_id = get_site_id_from_index(measurements.index)

    # keep all variables except age
    covars = get_volume_causes(clinical, standardize=False)
    covars.drop(["AGE", "I(AGE ** 2)"], axis=1, inplace=True)

    residuals = fit_combat(measurements, covars, site_id)

    data = pd.concat((residuals.add_suffix("-resid"), outcome), axis=1)
    out_file = output_dir / "adni_combat_residualized.csv"
    data.to_csv(out_file)

    return str(out_file.resolve())


def fit_model(input_csv: str, outputs_base_dir: str, random_seed: int = 2501) -> str:
    """Fit Beta regression model using rstanarm and store stan summary as CSV."""
    output_dir = Path(outputs_base_dir)
    input_csv_path = Path(input_csv)
    output_nb_file = output_dir / f"fit_{input_csv_path.stem}.ipynb"
    output_coef_file = output_dir / f"coef_{input_csv_path.stem}.csv"

    parameters = {
        "input_csv": input_csv,
        "coef_output_file": str(output_coef_file.resolve()),
        "rng_seed": random_seed,
    }

    pm.execute_notebook(
        Path(__file__).parent / "notebooks" / "adni_estimate_betareg.ipynb",
        output_nb_file,
        parameters=parameters,
        progress_bar=False,
        language="R",
    )

    opath = str(output_nb_file.resolve())
    LOG.info("Wrote output to %s", opath)

    return opath


def adni_fit_subst_conf_models(args):
    for inputs in create_substitute_confounder_inputs(
        args.features_dir, args.outcome_file, args.outputs_dir
    ):
        fit_model(inputs, args.outputs_dir)


def adni_fit_age_residualized_model(args):
    fit_model(
        create_age_residualized_inputs(args.outcome_file, args.outputs_dir),
        args.outputs_dir,
    )


def adni_fit_combat_residualized_model(args):
    fit_model(
        create_combat_residualized_inputs(args.outcome_file, args.outputs_dir),
        args.outputs_dir,
    )


def adni_fit_original_model(args):
    fit_model(
        create_original_inputs(args.outcome_file, args.outputs_dir),
        args.outputs_dir,
    )


def main(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser("subst_conf_models")
    p.add_argument("--outcome_file", required=True, help="Path to HDF5 with processed FreeSurfer features outcome data.")
    p.add_argument("--outputs_dir", required=True, help="Path to directory where to write outputs to.")
    p.add_argument("--features_dir", required=True, help="Path to directory where HDF5 files with features are located.")
    p.set_defaults(func=adni_fit_subst_conf_models)

    p = subparsers.add_parser("age_residualized_model")
    p.add_argument("--outcome_file", required=True, help="Path to HDF5 with processed FreeSurfer features outcome data.")
    p.add_argument("--outputs_dir", required=True, help="Path to directory where to write outputs to.")
    p.set_defaults(func=adni_fit_age_residualized_model)

    p = subparsers.add_parser("combat_residualized_model")
    p.add_argument("--outcome_file", required=True, help="Path to HDF5 with processed FreeSurfer features outcome data.")
    p.add_argument("--outputs_dir", required=True, help="Path to directory where to write outputs to.")
    p.set_defaults(func=adni_fit_combat_residualized_model)

    p = subparsers.add_parser("original_model")
    p.add_argument("--outcome_file", required=True, help="Path to HDF5 with processed FreeSurfer features outcome data.")
    p.add_argument("--outputs_dir", required=True, help="Path to directory where to write outputs to.")
    p.set_defaults(func=adni_fit_original_model)

    args = parser.parse_args(args=args)

    if not hasattr(args, "func"):
        parser.print_usage()
        return

    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
