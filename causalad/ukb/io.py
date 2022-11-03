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
from dataclasses import dataclass, fields
from typing import Dict

import numpy as np
import pandas as pd
import tables


@dataclass
class PatientData:
    volumes: pd.DataFrame
    thickness: pd.DataFrame
    demographics: pd.DataFrame
    confounders: pd.DataFrame


@dataclass
class SyntheticData:
    outcome: pd.Series
    coef: pd.Series
    confounders: pd.DataFrame


@dataclass
class SubstituteConfounderData:
    substitute_confounder: pd.DataFrame
    residuals: pd.DataFrame
    metadata: pd.DataFrame


def load_patient_data(data_path) -> PatientData:
    with pd.HDFStore(data_path, mode="r") as store:
        data = PatientData(
            volumes=store.get("volumes"),
            thickness=store.get("thickness"),
            demographics=store.get("demographics"),
            confounders=store.get("confounders"),
        )
    return data


def _write_dataclass(data, store, root="/"):
    for field in fields(data):
        values = getattr(data, field.name)
        key = f"{root}/{field.name}"
        store.put(key, values)


def write_synthetic_data(data: SyntheticData, path):
    with pd.HDFStore(path, mode="w", complib="lzo") as store:
        _write_dataclass(data, store)


def write_deconfounder_data(data: SubstituteConfounderData, posterior, path):
    with pd.HDFStore(path, mode="w", complib="lzo") as store:
        _write_dataclass(data, store)

    with tables.open_file(path, mode="a") as h5file:
        grp = h5file.create_group(h5file.root, "posterior", "Posterior Mean")
        for field in fields(posterior):
            values = getattr(posterior, field.name)
            h5file.create_array(grp, field.name, values)


def load_synthetic_data(path) -> SyntheticData:
    with pd.HDFStore(path, mode="r") as store:
        outcome = store.get("outcome").loc[:, "outcome"]
        coef = store.get("coef")
        confounders = store.get("confounders")
    return SyntheticData(outcome, coef, confounders)


def load_deconfounder_residuals(path) -> pd.DataFrame:
    with pd.HDFStore(path, mode="r") as store:
        features = store.get("residuals")
    return features


def read_deconfounder_posterior(path) -> Dict[str, np.ndarray]:
    posterior = {}
    with tables.open_file(path, mode="r") as h5file:
        for arr in h5file.root.posterior:
            posterior[arr.name] = arr.read()
    return posterior
