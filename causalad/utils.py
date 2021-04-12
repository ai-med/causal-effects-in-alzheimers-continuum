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

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger(__name__)


def drop_outliers(features: pd.DataFrame) -> pd.DataFrame:
    outliers = set()
    for col, vals in features.iteritems():
        lower, med, upper = np.percentile(vals, [25, 50, 75])
        scale = np.abs(upper - lower)
        outliers.update(vals[(vals < med - 3*scale) | (vals > med + 3*scale)].index)
    LOG.info("%d outliers removed", len(outliers))
    return features.drop(outliers, axis=0)


def boxcox(x, lmbda, mu, sigma):
    return (stats.boxcox(x, lmbda=lmbda) - mu) / sigma


def logt(x, mu, sigma):
    return (np.log1p(x) - mu) / sigma


def stdt(x, mu, sigma):
    return (x - mu) / sigma
