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
from scipy import special, stats
from sklearn.utils.validation import check_random_state

from .data import get_volume_causes
from .estimate import fit_regress_out
from .io import PatientData, SyntheticData


class ConfoundingGenerator:
    def __init__(
        self,
        data: PatientData,
        sparsity: float = 0.8,
        prob_event: float = 0.5,
        var_x: float = 0.4,
        var_z: float = 0.4,
        random_state: Optional[int] = None,
    ) -> None:
        assert var_x + var_z < 1.0
        self.data = data
        self.sparsity = sparsity
        self.prob_event = prob_event
        self.var_x = var_x
        self.var_z = var_z
        self.random_state = random_state

        self._site_id = None

    def logistic(self, eta):
        probs = special.expit(eta)
        probs = np.maximum(1e-5, np.minimum(1.0 - 1e-5, probs))
        outcome = stats.bernoulli.rvs(p=probs, random_state=self._rnd)
        return outcome

    def _make_sparse(self, x: np.ndarray) -> np.ndarray:
        """Only retain the largest coefficients"""
        o = np.negative(np.abs(x)).argsort()
        mask = o[:max(1, int(len(x) * self.sparsity))]
        x[mask] = 0.0
        return x

    def _get_volume_causes(self) -> pd.DataFrame:
        return get_volume_causes(self.data.demographics)

    def _generate_causal_effects(self, n_features):
        rv_x = stats.norm(loc=0.0, scale=0.5)
        coef_x = rv_x.rvs(size=n_features, random_state=self._rnd)
        coef_x = self._make_sparse(coef_x)
        return coef_x

    def _generate_unobserved_confounder(self):
        # confounding
        site_id = self.data.confounders.loc[:, "unobserved_confounder"]
        num_sites = site_id.nunique()

        # per group intercept and error variance
        intercepts = site_id.values
        # err_var = self._rnd.gamma(shape=8, scale=1./8, size=num_sites)
        err_var = stats.invgamma(a=3, loc=1).rvs(size=num_sites, random_state=self._rnd)
        print("Per-cluster intercepts: %s" % site_id.unique())
        print("Per-cluster error variance: %s" % err_var)

        return intercepts, err_var[site_id.values - 1]

    def _generate_observed_confounder(self):
        df = self.data.demographics.loc[:, ["AGE"]]
        rv = stats.norm(loc=0.0, scale=0.2)
        coef = rv.rvs(size=df.shape[1], random_state=self._rnd)

        mu = np.dot(df.values, coef)
        return mu

    def generate_outcome_with_site(self):
        # see https://github.com/blei-lab/deconfounder_public/blob/master/gene_tf/src/utils.py#L149
        self._rnd = check_random_state(self.random_state)
        var_noise = 1.0 - self.var_x - self.var_z
        assert var_noise > 0.0

        n_samples, n_features = self.data.volumes.shape
        n_features += self.data.thickness.shape[1]

        # causal effect
        obs_conf_data = self._get_volume_causes()

        vol_thick_data = pd.concat(
            (
                self.data.volumes,  # volumes have been divided by TIV in prepare_data already
                self.data.thickness,
            ),
            axis=1,
        )
        causal_data = fit_regress_out(vol_thick_data, obs_conf_data)

        # true causal effect
        coef_x = self._generate_causal_effects(n_features)
        mu_x = np.dot(causal_data.values, coef_x)
        std_x = np.std(mu_x, ddof=1)
        contrib_x = 1.0 / np.sqrt(self.var_x)

        # unobserved confounder
        mu_z_hidden, error_scale = self._generate_unobserved_confounder()
        mu_z_hidden_std = np.std(mu_z_hidden, ddof=1)

        contrib_z_hidden = contrib_x * np.sqrt(self.var_z * 0.9) / mu_z_hidden_std

        # observed confounder
        mu_z_obs = self._generate_observed_confounder()

        contrib_z_obs = contrib_x * np.sqrt(self.var_z * 0.1) / np.std(mu_z_obs, ddof=1)

        # noise
        noise = stats.norm.rvs(loc=np.zeros(n_samples), scale=error_scale, random_state=self._rnd)
        contrib_noise = contrib_x * np.sqrt(var_noise) / np.std(noise, ddof=1)

        mu = np.stack([mu_x, mu_z_hidden - mu_z_hidden.mean(), mu_z_obs, noise], axis=1)
        contrib = np.array([1.0 / std_x, contrib_z_hidden, contrib_z_obs, contrib_noise])

        # linear predictor
        eta = np.dot(mu, contrib)
        # center around 0
        eta -= np.mean(eta)

        intercept = stats.logistic.ppf(self.prob_event)
        eta += intercept

        true_coef = pd.DataFrame(coef_x / std_x, index=causal_data.columns, columns=["coefficient"])
        true_coef.loc["Intercept"] = intercept

        outcome = pd.DataFrame(
            self.logistic(eta),
            index=self.data.demographics.index,
            columns=["outcome"],
        )

        confounders = pd.concat(
            (self.data.confounders, self.data.demographics.loc[:, "AGE"]), axis=1
        )
        return SyntheticData(outcome, true_coef, confounders)
