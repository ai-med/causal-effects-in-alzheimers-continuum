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
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_adni_data(filename: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load data created by prepare_data.ipynb"""
    with pd.HDFStore(filename, mode="r") as store:
        volumes = store.get("volumes")
        thickness = store.get("thickness")
        clinical = store.get("clinical")
        outcome = store.get("outcome")

    measurements = pd.concat((volumes, thickness), axis=1)
    measurements, clinical = measurements.align(clinical, axis=0)
    outcome = outcome.loc[measurements.index]

    return measurements, clinical, outcome


def load_adni_data_full(filename: Path) -> Dict[str, pd.DataFrame]:
    """Load data created by prepare_data.ipynb"""
    data = {}
    with pd.HDFStore(filename, mode="r") as store:
        for path, _, leaves in store.walk():
            if len(path) == 0:
                for key in leaves:
                    data[key] = store.get(key)
                break

    return data
