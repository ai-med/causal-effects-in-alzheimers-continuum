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
from itertools import product
from typing import Dict, Callable, Tuple

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from .pystan_models.model import BPMF, LM, PPCA
from .utils import boxcox, drop_outliers, logt, stdt

LOG = logging.getLogger(__name__)

_ADNIMERGE_PATH = "data/adni-data.csv"
_APOE_PATH = "data/APOERES.csv"


class Volumes:
    THICKNESS = [
        'parahippocampal_thickness',
        'caudalanteriorcingulate_thickness',
        'fusiform_thickness',
        'lateralorbitofrontal_thickness',
        'insula_thickness',
        'entorhinal_thickness',
        'precuneus_thickness',
        'rostralmiddlefrontal_thickness',
    ]

    VOLUMES_LR = [
        'Cerebellum-White-Matter',
        'Cerebellum-Cortex',
        'Thalamus-Proper',
        'Caudate',
        'Putamen',
        'Pallidum',
        'Hippocampus',
        'Accumbens-area',
        'vessel',
    ]

    VOLUMES_VENTRICLE = [
        'Left-Lateral-Ventricle',
        'Left-Inf-Lat-Vent',
        'Right-Lateral-Ventricle',
        'Right-Inf-Lat-Vent',
        '3rd-Ventricle',
        '4th-Ventricle',
        '5th-Ventricle',
    ]

    VOLUMES_CC = [
        'CC_Posterior',
        'CC_Mid_Posterior',
        'CC_Central',
        'CC_Mid_Anterior',
        'CC_Anterior',
    ]

    VOLUMES_SINGLE = [
        'Brain-Stem',
        'CSF',
        'Optic-Chiasm',
        'eTIV',
    ]


def load_adnimerge(remove_outliers=False, outcome="ADAS13"):
    adni = pd.read_csv(_ADNIMERGE_PATH, low_memory=False)
    baseline = adni.sort_values(by=["PTID", "M"]).groupby("PTID").first().set_index("RID")
    assert baseline.index.is_unique

    baseline.loc[:, "PTGENDER"] = baseline.loc[:, "PTGENDER"]#.replace({"Female": 0, "Male": 1})

    baseline.loc[:, "EDU-ATTAIN"] = pd.cut(
        baseline.PTEDUCAT,
        bins=[0, 12, 16, np.infty],
        labels=["less_or_equal_12", "12-16", "more_than_16"],
        right=True,
    )
    LOG.info("\n%s\n", baseline.loc[:, "EDU-ATTAIN"].value_counts())

    log_cols = ["PTAU", "TAU"]
    for col, series in baseline.loc[:, log_cols].iteritems():
        baseline.loc[:, col] = np.log1p(series.values)

    features_sum = {}
    for col in Volumes.VOLUMES_LR:
        features_sum[col] = baseline.loc[:, [f"Left-{col}", f"Right-{col}"]].sum(axis=1)
    # Sum all CC volumes
    features_sum["CC"] = baseline.loc[:, Volumes.VOLUMES_CC].sum(axis=1)
    features_sum["Ventricle"] = baseline.loc[:, Volumes.VOLUMES_VENTRICLE].sum(axis=1)

    for col in Volumes.THICKNESS:
        features_sum[col] = baseline.loc[:, [f"lh_{col}", f"rh_{col}"]].mean(axis=1)
    features_sum = pd.DataFrame.from_dict(features_sum)

    mri_features = pd.concat((baseline.loc[:, Volumes.VOLUMES_SINGLE], features_sum), axis=1).dropna(axis=0)
    sd = mri_features.std(ddof=1)
    assert (sd > 1e-6).all(), "features with low variance:\n{}".format(sd[sd <= 1e-6])

    eTIV = mri_features.loc[:, "eTIV"]
    mri_features.drop("eTIV", axis=1, inplace=True)
    mri_features.loc[:, Volumes.VOLUMES_LR] = mri_features.loc[:, Volumes.VOLUMES_LR].div(eTIV, axis=0)

    if remove_outliers:
        mri_features = drop_outliers(mri_features)

    is_atn = baseline.loc[:, "ATN_status"].isin(
        ["A+/T+/N+", "A+/T+/N-", "A+/T-/N-"]
    )
    has_outcome = baseline.loc[:, outcome].notnull()
    positive_outcome = baseline.loc[:, outcome] > 0
    LOG.info("Dropping %d with missing or zero %s\n", baseline.shape[0] - positive_outcome.sum(), outcome)

    data = baseline.loc[is_atn & has_outcome & positive_outcome, :]

    y = data.loc[:, outcome].round(0).astype(int)
    LOG.info("\n%s\n", data.loc[:, "ATN_status"].value_counts())

    csf_features = ['ABETA', 'TAU', 'PTAU']

    demo_features = ['IMAGEUID', 'COLPROT', 'SITE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'EDU-ATTAIN']
    features = pd.concat((
        data.loc[:, demo_features],
        data.loc[:, "ATN_status"],
        data.loc[:, csf_features],
        mri_features
    ), axis=1, join="inner")

    assert features.notnull().all().all()
    assert y.notnull().all()

    return features, y


def load_apoe() -> pd.DataFrame:
    apoe = pd.read_csv(_APOE_PATH, usecols=["RID", "APGEN1", "APGEN2"], index_col="RID")
    assert apoe.index.is_unique

    ap1 = apoe.loc[:, "APGEN1"]
    ap2 = apoe.loc[:, "APGEN2"]

    LOG.info("\n%s\n", pd.crosstab(ap1, ap2))

    excluded = ((ap1 == 2) & (ap2 == 2)) | ((ap1 == 2) & (ap2 == 4))
    apoe.drop(excluded[excluded].index, axis=0, inplace=True)

    ap = pd.Series(index=apoe.index, name="ApoeSubtype", dtype=object)
    for a, b in product(apoe.APGEN1.unique(), apoe.APGEN2.unique()):
        mask = (apoe.APGEN1 == a) & (apoe.APGEN2 == b)
        ap.loc[mask] = f"Apo-e{a}e{b}"

    apoe.drop(["APGEN1", "APGEN2"], axis=1, inplace=True)
    apoe.loc[:, ap.name] = ap

    return apoe


def load_adni_data(**kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data, outcome = load_adnimerge(**kwargs)

    apoe = load_apoe()
    data = pd.concat((data, apoe), axis=1, join="inner")

    extra_features = [
        'IMAGEUID',
        'COLPROT',
        'SITE',
        'AGE',
        'PTGENDER',
        'PTEDUCAT',
        'EDU-ATTAIN',
        'ABETA',
        'TAU',
        'PTAU',
        'ATN_status',
        'ApoeSubtype',
    ]

    volumes = data.drop(extra_features, axis=1)
    others = data.loc[:, extra_features]
    return volumes, others, outcome.loc[data.index]


def get_volume_transforms(features: pd.DataFrame) -> Dict[str, Dict[str, Callable[[np.ndarray], np.ndarray]]]:
    log_transform = pd.Index([])

    bc_transform = pd.Index([
        "Accumbens-area",
        "Caudate",
        "Cerebellum-White-Matter",
        "CSF",
        "Hippocampus",
        "Pallidum",
        "Optic-Chiasm",
        "Putamen",
        "Ventricle",
        "vessel",
        "entorhinal_thickness",
    ])

    transforms = {}
    for col, vals in features.loc[:, bc_transform].iteritems():
        xt, lmbda = stats.boxcox(vals.values)
        mu = xt.mean()
        sd = xt.std(ddof=1)

        transforms[col] = partial(boxcox, lmbda=lmbda, mu=mu, sigma=sd)

    for col, vals in features.loc[:, log_transform].iteritems():
        xt = np.log1p(vals)
        mu = xt.mean()
        sd = xt.std(ddof=1)

        transforms[col] = partial(logt, mu=mu, sigma=sd)

    std_cols = features.columns.difference(bc_transform.union(log_transform))
    others = features.loc[:, std_cols]
    mus = others.mean()
    sds = others.std(ddof=1)
    for col, mu, sd in zip(std_cols, mus, sds):
        transforms[col] = partial(stdt, mu=mu, sigma=sd)

    return transforms


def apply_volume_transforms(data_fm: pd.DataFrame) -> pd.DataFrame:
    transform_fn = get_volume_transforms(data_fm)
    data_t = []
    for col, series in data_fm.iteritems():
        trans_fn = transform_fn[col]
        values_t = trans_fn(series.values)
        m = np.mean(values_t)
        s = np.std(values_t, ddof=1)
        assert s > 1e-8
        values_t -= m
        values_t /= s
        values_t = pd.Series(values_t, index=data_fm.index, name=col)
        data_t.append(values_t)
    return pd.concat(data_t, axis=1)


def get_regressed_out_volumes(
    data_volumes: pd.DataFrame,
    data_extra: pd.DataFrame,
    random_state: int
) -> pd.DataFrame:
    dm = patsy.dmatrix(
        "AGE + I(AGE**2) + Q('EDU-ATTAIN') + C(PTGENDER) - 1", data=data_extra, return_type="dataframe"
    )
    vol_causes = [
        "AGE",
        "I(AGE ** 2)",
        "Q('EDU-ATTAIN')[12-16]",
        "Q('EDU-ATTAIN')[more_than_16]",
        "C(PTGENDER)[T.Male]",
    ]
    dm = dm.loc[:, vol_causes]
    dm = (dm - dm.mean()) / dm.std(ddof=1)

    pystan_log = logging.getLogger("pystan")
    outputs = []
    for name, volume in data_volumes.iteritems():
        model = LM(random_state=random_state)
        try:
            lvl = pystan_log.level
            pystan_log.setLevel(logging.ERROR)
            model.fit(dm.values, volume.values, X_valid=dm.iloc[:1].values, y_valid=volume.iloc[:1].values)
        finally:
            pystan_log.setLevel(lvl)

        alphas = model.get_samples("alpha", False)
        betas = model.get_samples("beta")
        mus = alphas + np.dot(dm.values, betas)  # shape = [n_obervations, n_posterior_samples]
        vol_pred = np.mean(mus, axis=1)
        vol_resid = volume.values - vol_pred
        vol_resid = pd.Series(vol_resid, index=volume.index, name=name)
        outputs.append(vol_resid)

    outputs = pd.concat(outputs, axis=1).add_suffix("-resid")
    return outputs


def get_volume_causes(data_extra: pd.DataFrame) -> pd.DataFrame:
    dm = patsy.dmatrix(
        "AGE + I(AGE**2) + Q('EDU-ATTAIN') + C(PTGENDER) + PTAU - 1",
        data=data_extra, return_type="dataframe"
    )
    vol_causes = [
        "AGE",
        "I(AGE ** 2)",
        "Q('EDU-ATTAIN')[12-16]",
        "Q('EDU-ATTAIN')[more_than_16]",
        "C(PTGENDER)[T.Male]",
        "PTAU",
    ]
    dm = dm.loc[:, vol_causes]
    dm = (dm - dm.mean()) / dm.std(ddof=1)
    return dm


def get_ppca_deconfounder(
    data_volumes: pd.DataFrame,
    data_extra: pd.DataFrame,
    latent_dim: int,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    assert len(data_volumes.index.symmetric_difference(data_extra.index)) == 0

    dm = get_volume_causes(data_extra)
    vol_causes = dm.columns.tolist()
    dm = pd.concat((dm, data_volumes), axis=1)

    m = PPCA(known_causes=vol_causes, latent_dim=latent_dim, random_state=random_state)
    m.fit(dm)

    pval = m.check_model(silent=True)
    LOG.info("Overall p-value: %f", pval)

    posterior = m.posterior_mean_estimates()

    # approximate the (random variable) substitute confounders with their inferred mean.
    Z_hat = pd.DataFrame(posterior.Z_mu, index=dm.index).add_prefix("Deconf_")

    recon = m.mean_reconstruction(posterior.Z_mu)
    resid = data_volumes.values - recon
    output = pd.DataFrame(resid, index=dm.index, columns=data_volumes.columns).add_suffix("-resid")
    return Z_hat, output, pval


def get_bpmf_deconfounder(
    data_volumes: pd.DataFrame,
    data_extra: pd.DataFrame,
    latent_dim: int,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    dm = get_volume_causes(data_extra)
    vol_causes = dm.columns.tolist()
    dm = pd.concat((dm, data_volumes), axis=1)

    m = BPMF(vol_causes, latent_dim=latent_dim, random_state=random_state)
    m.fit(dm)

    pval = m.check_model(silent=True)
    LOG.info("Overall p-value: %f", pval)

    posterior = m.posterior_mean_estimates()

    # approximate the (random variable) substitute confounders with their inferred mean.
    Z_hat = pd.DataFrame(posterior.U, index=dm.index).add_prefix("Deconf_")

    recon = m.mean_reconstruction(posterior.U)
    resid = data_volumes.values - recon
    output = pd.DataFrame(resid, index=dm.index, columns=data_volumes.columns).add_suffix("-resid")
    return Z_hat, output, pval
