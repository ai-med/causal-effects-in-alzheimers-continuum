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
from dataclasses import dataclass
from itertools import groupby
import logging
from pathlib import Path
import pickle
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import papermill as pm
from tqdm import tqdm

from .configs import SimulationSettings, get_var_config

LOG = logging.getLogger(__name__)


@dataclass
class Experiment:
    cfg: SimulationSettings
    name: str
    files: List[str]


@dataclass
class ExperimentSummary:
    key: Tuple[str, str]
    value: float


def _settings_name(cfg: SimulationSettings):
    name = "x{x:d}_z{z:d}".format(
        x=cfg.ratio_x_z[0],
        z=cfg.ratio_x_z[1],
    )
    return name


def get_experiments(base_dir: Path) -> Iterable[Experiment]:
    configs = get_var_config()

    for cfg in configs:
        sim_dir = base_dir / "simulate_data_x{x:d}_z{z:d}_eps{eps:.4f}_target".format(
            x=cfg.ratio_x_z[0],
            z=cfg.ratio_x_z[1],
            eps=cfg.var_eps,
        )
        if not sim_dir.exists():
            LOG.warning(f"{sim_dir} does not exist")
            continue

        files = {}
        for seed_dir in sim_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            for afile in seed_dir.glob("eval_*"):
                files.setdefault(afile.name, []).append(str(afile.resolve()))

        for name, path in files.items():
            yield Experiment(cfg, name, path)


def load_bias(filename: str) -> pd.Series:
    with open(filename, "rb") as fin:
        df = pickle.load(fin)

    is_effect = df.loc[:, "coef_true"].notnull()  # only keep volume/thickness measures
    is_effect.loc["Intercept"] = False  # ignore intercept
    values = df.loc[is_effect, "bias"].copy()
    seed = Path(filename).parent.name
    values.rename(seed, inplace=True)
    return values


def concat_experiments(experiment: Experiment) -> pd.DataFrame:
    """Combine repeated experiments for one model into a single table"""
    df = pd.concat([load_bias(p) for p in experiment.files], axis=1)

    config_name = _settings_name(experiment.cfg)
    model_name = experiment.name
    df.columns = pd.MultiIndex.from_product(
        [[config_name], [model_name], df.columns.tolist()],
        names=["config", "model", "seed"],
    )
    return df


def save_all_experiments(output_path: Path, data: List[pd.DataFrame]) -> None:
    """Save estimated biases of all experiments in single HDF5 file"""
    # group by config
    def key_fn(x):
        return x.columns.levels[0][0]

    sorted_by_cfg = sorted(data, key=key_fn)
    with pd.HDFStore(output_path, mode="w") as store:
        for config, tables in groupby(sorted_by_cfg, key=key_fn):
            df = pd.concat(list(tables), axis=1)
            df.sort_index(axis=1, inplace=True)
            store.put(config, df)


def summarize_experiment(df: pd.DataFrame) -> ExperimentSummary:
    """Compute mean total bias across all repetitions"""
    df_sum = df.sum()  # sum over seeds, the columns
    return ExperimentSummary(
        tuple(k[0] for k in df.columns.levels[:2]), float(np.mean(df_sum.values))
    )


def _combine_summaries(results: List[ExperimentSummary]) -> pd.Series:
    index = np.empty(len(results), dtype=object)
    values = np.empty(len(results))
    for i, item in enumerate(results):
        index[i] = item.key[1]
        values[i] = item.value
    return pd.Series(values, index=index).sort_index()


def combine_all_summaries(output_path: Path, summaries: List[ExperimentSummary]) -> None:
    """Create and save table summarizing all experiments"""
    # group by config
    stats = {}
    for res in summaries:
        name = res.key[0]
        if name not in stats:
            stats[name] = []
        stats[name].append(res)

    df = {}
    for key, experiments in stats.items():
        df[key] = _combine_summaries(experiments)

    df = pd.DataFrame(df)
    df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Path to directory where sub-folders with different experiments are located.",
    )
    parser.add_argument(
        "--subst_conf_dir",
        type=Path,
        required=True,
        help="Directory containing HDF5 files with estimated substitute confounders.",
    )
    parser.add_argument(
        "--outputs_dir", type=Path, required=True, help="Path to directory where to write outputs to."
    )

    args = parser.parse_args()

    tables = []
    summaries = []
    n_repeats = None
    for experiment in tqdm(get_experiments(args.models_dir)):
        n_repeats = n_repeats or len(experiment.files)
        # if len(experiment.files) != n_repeats:
        #     raise AssertionError(
        #       f"expected {n_repeats}, but got {len(experiment.files)} for {experiment.name}"
        #     )

        table = concat_experiments(experiment)
        tables.append(table)
        summary = summarize_experiment(table)
        summaries.append(summary)

    out_dir = args.outputs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_all_experiments(out_dir / "all_experiments.h5", tables)
    summary_path = out_dir / "experiments_summary.csv"
    combine_all_summaries(summary_path, summaries)

    pm.execute_notebook(
        Path(__file__).parent / "notebooks" / "ukb_visualize.ipynb",
        out_dir / "ukb_visualize_output.ipynb",
        parameters={
            "subst_conf_dir": str(args.subst_conf_dir.resolve()),
            "bias_csv": str(summary_path.resolve()),
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
