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
from typing import Sequence, Tuple

import numpy as np


@dataclass
class SimulationSettings:
    """Settings for one simulation.

    Attributes
    ----------
    seed : int
        Random number seed.
    var_x : float
        Variance of direct effect.
    var_z : float
        Variance of indirect effect.
    var_eps : float
        Variance of noise term.
    ratio_x_z : (int, int)
        Integer values for ``var_x`` and ``var_z``.
    """
    seed: int
    var_x: float
    var_z: float
    var_eps: float
    ratio_x_z: Tuple[int, int]

def get_var_config(
    noise_var: float = 0.2,
    n_repeats: int = 1,
    random_state: int = 1802080521,
) -> Sequence[SimulationSettings]:
    """Returns different settings for simulations.

    .. math::

        \begin{aligned}
            \mathrm{Var}(\mu) &= \nu_x + \nu_z + \nu_\varepsilon = 1 \\
            \frac{\nu_x}{\nu_z} &= r \\
            1 - \nu_\varepsilon &= \nu_x + \nu_z = \nu_x + \frac{\nu_x}{r}  = \nu_x \frac{r + 1}{r} \\
            &= r \nu_z + \nu_z = \nu_z (r + 1) \\
            \Leftrightarrow \nu_x &= (1 - \nu_\varepsilon) \frac{r}{r + 1}  \\
            \Leftrightarrow \nu_z &= \frac{1 - \nu_\varepsilon}{r + 1}
        \end{aligned}

    Parameters
    ----------
    noise_var : float
        Variance of unexplained noise.

    n_repeats : int
        Number of times to repeat each experiment.

    random_state : int
        Random number seed.

    Returns
    -------
    settings : list of :class:`SimulationSettings`.
        The list of simulation configs.
    """
    assert noise_var > 0.
    assert noise_var < 1.

    a = np.array([2, 3, 2, 1, 1, 1, 1], dtype=int)
    b = np.array([3, 5, 5, 3, 4, 5, 10], dtype=int)

    nom = np.concatenate(([1], a, b))
    denom = np.concatenate(([1], b, a))

    r = nom / denom

    var_left = 1.0 - noise_var
    var_causal = var_left * r / (r + 1.0)
    var_confounded = var_left / (r + 1.0)

    rnd = np.random.RandomState(random_state)
    seeds = rnd.randint(np.iinfo(np.int32).max, size=n_repeats)

    settings = [
        SimulationSettings(seed, var_x, var_z, noise_var, (r_nom, r_denom))
        for seed, (var_x, var_z, r_nom, r_denom) in product(
            map(int, seeds),
            zip(var_causal, var_confounded, nom, denom),
        )
    ]
    return settings
