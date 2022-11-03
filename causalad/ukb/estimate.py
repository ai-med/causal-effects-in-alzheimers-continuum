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
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..pystan_models.models import LogReg, RandomStateType
from .data import get_volume_causes


def fit_regress_out(X: pd.DataFrame, confounders: pd.DataFrame) -> pd.DataFrame:
    X, conf = X.align(confounders, axis=0, join="inner")
    conf = conf.values

    est = LinearRegression()
    X_resid = np.empty(X.shape, dtype=float, order="F")
    for i, (_, col) in enumerate(X.iteritems()):
        est.fit(conf, col.values)
        X_resid[:, i] = col.values - est.predict(conf)

    X_resid = pd.DataFrame(X_resid, index=X.index, columns=X.columns, copy=False)

    return X_resid


class OutcomeModelFitter:
    def __init__(
        self,
        scale_alpha: float = 5.0,
        scale_beta: float = 5.0,
        standardize: bool = True,
        use_regress_out: bool = True,
        posterior_samples: int = 1000,
        n_chains: int = 4,
        random_state: RandomStateType = None,
        n_jobs: int = 1,
    ) -> None:
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.standardize = standardize
        self.use_regress_out = use_regress_out
        self.posterior_samples = posterior_samples
        self.n_chains = n_chains
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_standard(self, features: pd.DataFrame, outcome: pd.Series) -> pd.DataFrame:
        X, y = features.align(outcome, axis=0)

        model = LogReg(
            scale_alpha=self.scale_alpha,
            scale_beta=self.scale_beta,
            standardize=self.standardize,
            posterior_samples=self.posterior_samples,
            n_chains=self.n_chains,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        model.fit(X.values, y.values)

        samples = model.get_posterior_samples(names=X.columns)
        return samples

    def fit_oracle(
        self,
        features: pd.DataFrame,
        outcome: pd.Series,
        demographics: pd.DataFrame,
        unobserved_confounder: pd.Series,
    ) -> pd.DataFrame:
        feat_causes = get_volume_causes(demographics)
        if self.use_regress_out:
            feat_residuals = fit_regress_out(features, feat_causes)

            features_oracle = pd.concat((feat_residuals, unobserved_confounder), axis=1, join="inner")
        else:
            features_oracle = pd.concat((features, feat_causes, unobserved_confounder), axis=1, join="inner")

        return self.fit_standard(features_oracle, outcome)

    def fit_observed_confounders(
        self,
        features: pd.DataFrame,
        outcome: pd.Series,
        demographics: pd.DataFrame,
    ) -> pd.DataFrame:
        feat_causes = get_volume_causes(demographics)
        if self.use_regress_out:
            feat_residuals = fit_regress_out(features, feat_causes)
        else:
            feat_residuals = pd.concat((features, feat_causes), axis=1, join="inner")

        return self.fit_standard(feat_residuals, outcome)

    def fit_substitute_confounders(
        self,
        residuals_or_features: pd.DataFrame,
        outcome: pd.Series,
        demographics: Optional[pd.DataFrame] = None,
        substitute_confounders: Optional[pd.DataFrame] = None,
    ):
        if self.use_regress_out:
            features = residuals_or_features
        else:
            feat_causes = get_volume_causes(demographics)
            features = pd.concat((
                residuals_or_features,
                feat_causes,
                substitute_confounders,
            ), axis=1)

        return self.fit_standard(features, outcome)
