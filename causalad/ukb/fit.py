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
from dataclasses import dataclass
from itertools import product
import logging
from pathlib import Path
import pickle
from typing import Iterable, Optional, Tuple

import pandas as pd

from .estimate import OutcomeModelFitter
from .evaluate import evaluate
from .io import load_deconfounder_residuals, load_patient_data, load_synthetic_data

RANDOM_STATE = 52711171

LOG = logging.getLogger(__name__)


@dataclass
class ModelInput:
    features_path: Path
    outcome_path: Path
    output_dir: str
    data_file: Optional[str] = None


def _get_generated_data(base_dir: str) -> Iterable[Path]:
    import os

    base_dir = Path(base_dir)
    for sim_dir in base_dir.glob("simulate_data_*"):
        for root, _, filenames in os.walk(sim_dir):
            for f in filenames:
                if f == 'data_generated.h5':
                    yield Path(root) / f


def _remove_suffix(x):
    if x.endswith("-resid"):
        return x[:-6]
    return x


def _name_coef_axis(inputs, coef_true, coef_est, prefix):
    coef_est.rename(_remove_suffix, axis=0, inplace=True)
    coef_true = coef_true.rename_axis(inputs.output_dir)
    coef_est = coef_est.rename_axis(f"{prefix}{inputs.features_path.stem}")
    return (coef_true, coef_est)


def _write_coefficients(
    base_dir: Path, output_name: str, outputs: Tuple[pd.DataFrame, pd.DataFrame],
) -> Path:
    coef_est = outputs[1]
    out_file = base_dir / output_name / f"coefs_{coef_est.index.name}"
    LOG.info("Writing coefficients to %s", out_file)
    with open(out_file, "wb") as fout:
        pickle.dump(outputs, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return out_file


def _get_substitute_input_data(
    base_dir_path: Path, features_dir: Path, data_file: Optional[Path] = None,
) -> Iterable[ModelInput]:
    for data_path, sim_path in product(
        features_dir.glob("subst_conf_*.h5"),
        _get_generated_data(base_dir_path)
    ):
        out_dir = str(sim_path.parent.relative_to(base_dir_path))
        result_file = sim_path.with_name(f"coefs_{data_path.stem}")
        if not result_file.exists():
            data = ModelInput(data_path, sim_path, out_dir, data_file)
            yield data


def fit_substitute_model(inputs: ModelInput) -> Tuple[pd.DataFrame, pd.DataFrame]:
    use_regress_out = pd.isnull(inputs.data_file)

    synthetic_data = load_synthetic_data(inputs.outcome_path)
    if use_regress_out:
        features = load_deconfounder_residuals(inputs.features_path)
        demographics = None
        subst_conf = None
    else:
        patient_data = load_patient_data(inputs.data_file)
        features = pd.concat((patient_data.volumes, patient_data.thickness), axis=1)
        subst_conf = pd.read_hdf(inputs.features_path, key="substitute_confounder")
        demographics = patient_data.demographics

    fitter = OutcomeModelFitter(
        scale_beta=1.0,
        use_regress_out=use_regress_out,
        posterior_samples=1500,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    coef_est = fitter.fit_substitute_confounders(
        features,
        synthetic_data.outcome,
        demographics=demographics,
        substitute_confounders=subst_conf,
    )

    outputs = _name_coef_axis(inputs, synthetic_data.coef, coef_est, prefix="")

    return outputs


def eval_estimates(base_dir: Path, coefs: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    out_file = base_dir / f"{coefs[0].index.name}" / f"eval_{coefs[1].index.name}"

    if not out_file.exists():
        coef_true, coef_est = coefs
        coef_true = coef_true.loc[:, "coefficient"].drop("Intercept")
        data = evaluate(coef_true, coef_est)
        data.index.name = "+".join([coef_true.index.name, coef_est.index.name])

        LOG.info("Writing result to %s", out_file)
        with open(out_file, "wb") as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        LOG.info("Result %s already exists", out_file)


def fit_and_eval_substitute_confounders(args):
    for cfg in _get_substitute_input_data(args.models_dir, args.features_dir, args.data_file):
        coefs = fit_substitute_model(cfg)
        out_file = _write_coefficients(args.models_dir, cfg.output_dir, coefs)
        eval_estimates(args.models_dir, coefs)


def _get_input_data(
    base_dir_path: Path, data_path: Path, result_prefix: str,
) -> Iterable[ModelInput]:
    for sim_path in _get_generated_data(base_dir_path):
        out_dir = str(sim_path.parent.relative_to(base_dir_path))
        result_file = sim_path.with_name(f"coefs_{result_prefix}_{data_path.stem}")
        if not result_file.exists():
            data = ModelInput(data_path, sim_path, out_dir)
            yield data


def _get_fitter():
    fitter = OutcomeModelFitter(
        scale_beta=1.0,
        use_regress_out=False,
        posterior_samples=1500,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    return fitter


def fit_oracle_model(inputs: ModelInput) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patient_data = load_patient_data(inputs.features_path)
    synthetic_data = load_synthetic_data(inputs.outcome_path)

    features = pd.concat((patient_data.volumes, patient_data.thickness), axis=1)

    fitter = _get_fitter()
    coef_est = fitter.fit_oracle(
        features=features,
        outcome=synthetic_data.outcome,
        demographics=patient_data.demographics,
        unobserved_confounder=synthetic_data.confounders.loc[:, "unobserved_confounder"],
    )

    outputs = _name_coef_axis(inputs, synthetic_data.coef, coef_est, prefix="oracle_")
    return outputs


def fit_and_eval_oracle_model(args):
    for cfg in _get_input_data(args.models_dir, args.data_file, result_prefix="oracle"):
        coefs = fit_oracle_model(cfg)
        out_file = _write_coefficients(args.models_dir, cfg.output_dir, coefs)
        eval_estimates(args.models_dir, coefs)


def fit_standard_model(inputs: ModelInput) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patient_data = load_patient_data(inputs.features_path)
    synthetic_data = load_synthetic_data(inputs.outcome_path)

    features = pd.concat((patient_data.volumes, patient_data.thickness), axis=1)

    fitter = _get_fitter()
    coef_est = fitter.fit_standard(
        features=features,
        outcome=synthetic_data.outcome,
    )

    outputs = _name_coef_axis(inputs, synthetic_data.coef, coef_est, prefix="noconf_")
    return outputs


def fit_and_eval_no_confounders(args):
    for cfg in _get_input_data(args.models_dir, args.data_file, result_prefix="noconf"):
        coefs = fit_standard_model(cfg)
        out_file = _write_coefficients(args.models_dir, cfg.output_dir, coefs)
        eval_estimates(args.models_dir, coefs)


def fit_observed_model(inputs: ModelInput) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patient_data = load_patient_data(inputs.features_path)
    synthetic_data = load_synthetic_data(inputs.outcome_path)

    features = pd.concat((patient_data.volumes, patient_data.thickness), axis=1)

    fitter = _get_fitter()
    coef_est = fitter.fit_observed_confounders(
        features=features,
        outcome=synthetic_data.outcome,
        demographics=patient_data.demographics,
    )

    outputs = _name_coef_axis(inputs, synthetic_data.coef, coef_est, prefix="obsconf_")
    return outputs


def fit_and_eval_observed_confounders(args):
    for cfg in _get_input_data(args.models_dir, args.data_file, result_prefix="obsconf"):
        coefs = fit_observed_model(cfg)
        out_file = _write_coefficients(args.models_dir, cfg.output_dir, coefs)
        eval_estimates(args.models_dir, coefs)


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser("subst_conf_models")
    p.add_argument(
        "--features_dir", type=Path, required=True, help="Path to directory where HDF5 files with substitute confounder are located."
    )
    p.add_argument(
        "--models_dir", type=Path, required=True, help="Path to directory containing fitted models."
    )
    p.add_argument(
        "--data_file", type=Path, help="Path to HDF5 file containing original data."
    )
    p.set_defaults(func=fit_and_eval_substitute_confounders)

    p = subparsers.add_parser("oracle_model")
    p.add_argument(
        "--data_file", type=Path, required=True, help="Path to HDF5 file containing original data."
    )
    p.add_argument(
        "--models_dir", type=Path, required=True, help="Path to directory containing fitted models."
    )
    p.set_defaults(func=fit_and_eval_oracle_model)

    p = subparsers.add_parser("no_confounders")
    p.add_argument(
        "--data_file", type=Path, required=True, help="Path to HDF5 file containing original data."
    )
    p.add_argument(
        "--models_dir", type=Path, required=True, help="Path to directory containing fitted models."
    )
    p.set_defaults(func=fit_and_eval_no_confounders)

    p = subparsers.add_parser("observed_confounders")
    p.add_argument(
        "--data_file", type=Path, required=True, help="Path to HDF5 file containing original data."
    )
    p.add_argument(
        "--models_dir", type=Path, required=True, help="Path to directory containing fitted models."
    )
    p.set_defaults(func=fit_and_eval_observed_confounders)

    args = parser.parse_args(args=args)

    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
