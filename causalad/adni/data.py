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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import patsy

from ..utils import apply_box_cox_and_standardize, drop_outliers

LOG = logging.getLogger(__name__)


def get_site_id_from_index(index: pd.Index) -> pd.Series:
    site_id = index.to_series().str.split(
        "_", expand=True
    ).iloc[:, 0].astype(int).rename("SITE_ID")
    return site_id


class AdniDataLoader:
    def __init__(self, csv_file: str, drop_outliers: bool = True) -> None:
        self._csv_file = csv_file
        self._drop_outliers = drop_outliers
        self._outcome = "ADAS13"
        self._include_visit = "bl"
        self._index_col = ["PTID", "VISCODE"]
        self._demo_features = ['IMAGEUID', 'COLPROT', 'SITE', 'AGE', 'PTGENDER', 'PTEDUCAT']
        self._csf_features = ['ABETA', 'TAU', 'PTAU']

    def load_freesurfer(self) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
        all_features = self._load_csv()

        outcome = self._process_outcome(all_features)
        atn_status = self._process_atn(all_features)
        clinical = self._process_clinical(all_features)

        volumes = self._process_volumes(all_features)
        thickness = self._process_thickness(all_features)
        if self._drop_outliers:
            volumes = drop_outliers(volumes)
            thickness = drop_outliers(thickness)

        features = pd.concat((
            clinical,
            atn_status,
            volumes,
            thickness,
            all_features.loc[:, 'eTIV'],
        ), axis=1, join="inner")
        features, outcome = features.align(outcome, axis=0, join="inner")

        features = {
            "volumes": features.loc[:, volumes.columns],
            "thickness": features.loc[:, thickness.columns],
            "tiv": features.loc[:, 'eTIV':'eTIV'],
            "clinical": features.loc[:, clinical.columns.union({atn_status.name}, sort=False)]
        }

        return features, outcome

    def _load_csv(self):
        all_data = pd.read_csv(self._csv_file, index_col=self._index_col)
        all_features = all_data.xs(self._include_visit, level=1)
        assert all_features.index.is_unique
        return all_features

    def _process_volumes(self, all_features: pd.DataFrame) -> pd.DataFrame:
        # https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg07933.html
        ventricle_cols = [
            'Left-Lateral-Ventricle',
            'Left-Inf-Lat-Vent',
            'Right-Lateral-Ventricle',
            'Right-Inf-Lat-Vent',
            '3rd-Ventricle',
            '4th-Ventricle',
            '5th-Ventricle',
            'CSF',
            'Left-choroid-plexus',
            'Right-choroid-plexus',
        ]
        ventricles = all_features.loc[:, ventricle_cols].sum(axis=1).rename("Total_Ventricular_CSF")

        # corpus calosum
        cc_cols = [
            'CC_Posterior',
            'CC_Mid_Posterior',
            'CC_Central',
            'CC_Mid_Anterior',
            'CC_Anterior',
        ]
        cc = all_features.loc[:, cc_cols].sum(axis=1).rename("CC")

        volume_cols = [
            'Left-Cerebellum-Cortex',
            'Left-Thalamus-Proper',
            'Left-Caudate',
            'Left-Putamen',
            'Left-Pallidum',
            'Brain-Stem',
            'Left-Hippocampus',
            'Left-Amygdala',
            'Left-Accumbens-area',
            'Right-Cerebellum-Cortex',
            'Right-Thalamus-Proper',
            'Right-Caudate',
            'Right-Putamen',
            'Right-Pallidum',
            'Right-Hippocampus',
            'Right-Amygdala',
            'Right-Accumbens-area',
            'Optic-Chiasm',
        ]

        # Sum up measurements of left and right hemisphere.
        column_pairs = {}
        for col in volume_cols:
            for prefix in ("Left-", "Right-",):
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    column_pairs.setdefault(suffix, []).append(col)
                    break

        paired_data = {}
        for name, pair in column_pairs.items():
            assert len(pair) == 2
            paired_data[name] = all_features.loc[:, pair].sum(axis=1)
        paired_data = pd.DataFrame.from_dict(paired_data)

        features = pd.concat(
            (paired_data, cc, ventricles, all_features.loc[:, ['Brain-Stem', 'Optic-Chiasm']]),
        axis=1)

        return features

    def _process_thickness(self, all_features: pd.DataFrame) -> pd.DataFrame:
        thickness_cols = [
            'bankssts_thickness',
            'caudalanteriorcingulate_thickness',
            'caudalmiddlefrontal_thickness',
            'cuneus_thickness',
            'entorhinal_thickness',
            'fusiform_thickness',
            'inferiorparietal_thickness',
            'inferiortemporal_thickness',
            'isthmuscingulate_thickness',
            'lateraloccipital_thickness',
            'lateralorbitofrontal_thickness',
            'lingual_thickness',
            'medialorbitofrontal_thickness',
            'middletemporal_thickness',
            'parahippocampal_thickness',
            'paracentral_thickness',
            'parsopercularis_thickness',
            'parsorbitalis_thickness',
            'parstriangularis_thickness',
            'pericalcarine_thickness',
            'postcentral_thickness',
            'posteriorcingulate_thickness',
            'precentral_thickness',
            'precuneus_thickness',
            'rostralanteriorcingulate_thickness',
            'rostralmiddlefrontal_thickness',
            'superiorfrontal_thickness',
            'superiorparietal_thickness',
            'superiortemporal_thickness',
            'supramarginal_thickness',
            'frontalpole_thickness',
            'temporalpole_thickness',
            'transversetemporal_thickness',
            'insula_thickness',
        ]

        thick_avg = pd.DataFrame(index=all_features.index, columns=thickness_cols, dtype=float)
        for col in thickness_cols:
            thick_avg.loc[:, col] = all_features.loc[:, [f"lh_{col}", f"rh_{col}"]].mean(axis=1)

        return thick_avg

    def _process_edu(self, all_features: pd.DataFrame):
        edu = pd.cut(
            all_features.loc[:, "PTEDUCAT"],
            bins=[0, 12, 16, np.infty],
            labels=["less_or_equal_12", "12-16", "more_than_16"],
            right=True,
        ).rename('EDU-ATTAIN')
        return edu

    def _process_clinical(self, all_features: pd.DataFrame) -> pd.DataFrame:
        all_features = all_features.reset_index().set_index("RID")
        edu = self._process_edu(all_features)

        data = pd.concat(
            (
                all_features.loc[:, "PTID"],
                all_features.loc[:, self._demo_features],
                edu,
                all_features.loc[:, self._csf_features],
            ),
            axis=1,
            join="inner",
        ).set_index("PTID")
        return data

    def _process_outcome(self, all_features: pd.DataFrame) -> pd.Series:
        outcome = self._outcome
        has_outcome = all_features.loc[:, outcome].notnull()
        positive_outcome = all_features.loc[:, outcome] > 0
        valid_outcome = has_outcome & positive_outcome

        n_before = all_features.shape[0]
        n_after = valid_outcome.sum()
        LOG.info("Dropping %d with missing or zero %s\n", n_before - n_after, outcome)

        y = all_features.loc[valid_outcome, outcome].round(0).astype(int)
        return y

    def _process_atn(self, all_features: pd.DataFrame) -> pd.DataFrame:
        is_amyloid_positive = all_features.loc[:, "ATN_status"].isin(
            # ["A-/T-/N-", "A+/T+/N-", "A+/T+/N+"]
            ["A+/T+/N+", "A+/T+/N-", "A+/T-/N-"]
        )
        status = all_features.loc[is_amyloid_positive, "ATN_status"]
        LOG.info("\n%s\n", status.value_counts())

        return status


def apply_transform(features: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    # normalize volumes by dividing by eTIV
    tiv = features["tiv"].loc[:, "eTIV"]
    vols_t = features["volumes"].copy()
    vols_t = vols_t.div(tiv, axis=0)

    vols_t, transforms = apply_box_cox_and_standardize(vols_t)
    thicks_t, transforms_thicks = apply_box_cox_and_standardize(features["thickness"])
    transforms = pd.concat((transforms, transforms_thicks))

    clinical_t = features["clinical"].copy()
    for col in ("TAU", "PTAU",):
        clinical_t.loc[:, col] = np.log1p(clinical_t.loc[:, col].values)

    features_t = {
        "volumes": vols_t,
        "thickness": thicks_t,
        "tiv": features["tiv"],
        "clinical": clinical_t,
    }

    return features_t, transforms


def get_volume_causes(data_extra: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return VolumeCauseTransform(**kwargs).fit_transform(data_extra)


class VolumeCauseTransform:
    def __init__(self, standardize: bool = True) -> None:
        self.standardize = standardize
        self._formula = "AGE + I(AGE**2) + Q('EDU-ATTAIN') + C(PTGENDER) + PTAU - 1"
        self._columns = [
            "AGE",
            "I(AGE ** 2)",
            "Q('EDU-ATTAIN')[12-16]",
            "Q('EDU-ATTAIN')[more_than_16]",
            "C(PTGENDER)[T.Male]",
            "PTAU",
        ]

    def fit(self, data):
        dm = patsy.dmatrix(self._formula, data=data, return_type="dataframe")
        self._design_info = dm.design_info
        dm = dm.loc[:, self._columns]
        if self.standardize:
            self._design_mean = dm.mean()
            self._design_std = dm.std(ddof=1)
        else:
            self._design_mean = pd.Series(0.0, index=dm.columns)
            self._design_std = pd.Series(1.0, index=dm.columns)
        return self

    def _standardize(self, data):
        return (data - self._design_mean) / self._design_std

    def fit_transform(self, data):
        dm = patsy.dmatrix(self._formula, data=data, return_type="dataframe")
        self._design_info = dm.design_info

        dm = dm.loc[:, self._columns]
        self._design_mean = dm.mean()
        self._design_std = dm.std(ddof=1)

        return self._standardize(dm)

    def transform(self, new_data):
        mat = patsy.dmatrix(self._design_info, new_data, return_type="dataframe")
        dm = mat.loc[:, self._columns]

        return self._standardize(dm)
