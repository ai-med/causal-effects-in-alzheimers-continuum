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
import pandas as pd
from neuroCombat import neuroCombat


def fit_combat(
    features: pd.DataFrame,
    keep_vars: pd.DataFrame,
    site_id: pd.Series,
) -> pd.DataFrame:
    categorical_cols = ["C(PTGENDER)[T.Male]"]
    continuous_cols = keep_vars.columns.difference(categorical_cols).tolist()
    covars = pd.concat((keep_vars, site_id), axis=1)
    assert covars.columns.is_unique

    result_combat = neuroCombat(
        dat=features.T,  # shape = (features, samples)
        covars=covars,   # shape = (samples, covariates)
        batch_col=site_id.name,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )

    residuals = pd.DataFrame(
        result_combat["data"].T,
        index=features.index,
        columns=features.columns
    )
    return residuals
