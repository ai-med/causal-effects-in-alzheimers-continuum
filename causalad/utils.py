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
from functools import partial
from typing import Tuple

import pandas as pd
import numpy as np
from scipy import stats

LOG = logging.getLogger(__name__)


def drop_outliers(features: pd.DataFrame) -> pd.DataFrame:
    outliers = set()
    for col, vals in features.iteritems():
        is_zero = vals < np.finfo(np.float64).eps
        lower, med, upper = np.percentile(vals, [25, 50, 75])
        scale = np.abs(upper - lower)
        outliers.update(vals[(vals < med - 3*scale) | (vals > med + 3*scale) | is_zero].index)
    LOG.info("%d outliers removed", len(outliers))
    return features.drop(outliers, axis=0)


def apply_box_cox_and_standardize(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_t = {}
    transforms = pd.DataFrame(index=data.columns, columns=["mean", "stddev", "lmbda"], dtype=float)
    for name, col in data.iteritems():
        col_t, lmb = stats.boxcox(col.values)
        data_t[name] = col_t
        transforms.loc[name, "lmbda"] = lmb

    data_t = pd.DataFrame.from_dict(data_t)
    data_t.index = data.index

    # standardize
    m = data_t.mean()
    sd = data_t.std(ddof=1)
    data_t -= m
    data_t /= sd

    transforms.loc[:, "mean"] = m
    transforms.loc[:, "stddev"] = sd

    return data_t, transforms


def boxcox(x, lmbda, mu, sigma):
    return (stats.boxcox(x, lmbda=lmbda) - mu) / sigma


def boxcox_inv(y, lmbda, mu, sigma):
    return np.power((y * sigma + mu) * lmbda + 1, 1. / lmbda)


def boxcox_deriv(x, lmbda, mu, sigma):
    return np.power(x, lmbda - 1) / sigma


def logt(x, mu, sigma):
    return (np.log1p(x) - mu) / sigma


def logt_inv(y, mu, sigma):
    return np.exp(y * sigma + mu) - 1


def logt_deriv(x, mu, sigma):
    return np.reciprocal(sigma * (1 + x))


def stdt(x, mu, sigma):
    return (x - mu) / sigma


def stdt_inv(y, mu, sigma):
    return y * sigma + mu


def stdt_deriv(x, mu, sigma):
    return 1.0 / sigma


def get_transforms(features, log_transform, bc_transform=None):
    if log_transform is None:
        log_transform = pd.Index([])
    if bc_transform is None:
        bc_transform = pd.Index([])

    transforms = {}
    for col, vals in features.loc[:, bc_transform].iteritems():
        xt, lmbda = stats.boxcox(vals.values)
        mu = xt.mean()
        sd = xt.std(ddof=1)

        transforms[col] = {"t_1": partial(boxcox, lmbda=lmbda, mu=mu, sigma=sd),
                           "t": partial(boxcox_inv, lmbda=lmbda, mu=mu, sigma=sd),
                           "dt_1": partial(boxcox_deriv, lmbda=lmbda, mu=mu, sigma=sd)}

    for col, vals in features.loc[:, log_transform].iteritems():
        xt = np.log1p(vals)
        mu = xt.mean()
        sd = xt.std(ddof=1)

        transforms[col] = {"t_1": partial(logt, mu=mu, sigma=sd),
                           "t": partial(logt_inv, mu=mu, sigma=sd),
                           "dt_1": partial(logt_deriv, mu=mu, sigma=sd)}

    std_cols = features.columns.difference(bc_transform.union(log_transform))
    others = features.loc[:, std_cols]
    mus = others.mean()
    sds = others.std(ddof=1)
    for col, mu, sd in zip(std_cols, mus, sds):
        transforms[col] = {"t_1": partial(stdt, mu=mu, sigma=sd),
                           "t": partial(stdt_inv, mu=mu, sigma=sd),
                           "dt_1": partial(stdt_deriv, mu=mu, sigma=sd)}
    return transforms
