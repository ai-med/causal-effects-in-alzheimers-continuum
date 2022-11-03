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


def evaluate(coef_true: pd.Series, coef_est_samples: pd.DataFrame) -> pd.DataFrame:
    """Compute Bias, Variance, and Mean Sqaured Error of estimates.

    .. math::

        \begin{aligned}
            \operatorname {MSE} ({\hat {\theta }})
            &=\mathbb {E} [({\hat {\theta }}-\theta )^{2}] \\
            &=\mathbb {E} ({\hat {\theta }}^{2})+\mathbb {E} (\theta ^{2})
                - 2\theta \mathbb {E} ({\hat {\theta }})\\
            &=\operatorname {Var} ({\hat {\theta }})+(\mathbb {E} {\hat {\theta }})^{2}
                + \theta ^{2}-2\theta \mathbb {E} ({\hat {\theta }})\\
            &=\operatorname {Var} ({\hat {\theta }})+(\mathbb {E} {\hat {\theta }}-\theta )^{2}\\
            &=\operatorname {Var} ({\hat {\theta }})+\operatorname {Bias} ^{2}({\hat {\theta }})
        \end{aligned}

    Parameters
    ----------
    coef_true : pd.Series, shape=(n_features,)
        The true value of coefficients.

    coef_est_samples : pd.DataFrame, shape=(n_features, n_samples)
        Posterior samples of estimated coefficients.

    See Also
    --------
    - https://en.wikipedia.org/wiki/Mean_squared_error#Estimator
    - https://github.com/blei-lab/deconfounder_public/blob/c315808fda409389750ecc63acc10282bbc6305e/smoking_R/src/utils.R#L38
    """
    coef_est_mean = coef_est_samples.mean(axis=1)
    coef_bias = (coef_est_mean - coef_true).pow(2.0)
    coef_var = coef_est_samples.var(axis=1)
    mse = coef_var + coef_bias
    return pd.DataFrame(
        {
            "coef_true": coef_true,
            "coef_est_mean": coef_est_mean,
            "bias": coef_bias,
            "var": coef_var,
            "mse": mse
        }
    )
