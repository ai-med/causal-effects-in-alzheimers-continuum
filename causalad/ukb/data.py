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
from io import StringIO
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import patsy
import seaborn as sns

from ..utils import apply_box_cox_and_standardize, drop_outliers

LOG = logging.getLogger(__name__)

_LOBE_MAPPING = """Measurement	Lobe
caudalanteriorcingulate_thickness	Limbic
isthmuscingulate_thickness	Limbic
posteriorcingulate_thickness	Limbic
rostralanteriorcingulate_thickness	Limbic
parahippocampal_thickness	Limbic
caudalmiddlefrontal_thickness	Frontal
frontalpole_thickness	Frontal
lateralorbitofrontal_thickness	Frontal
medialorbitofrontal_thickness	Frontal
parsopercularis_thickness	Frontal
parsorbitalis_thickness	Frontal
parstriangularis_thickness	Frontal
precentral_thickness	Frontal
rostralmiddlefrontal_thickness	Frontal
superiorfrontal_thickness	Frontal
cuneus_thickness	Occipital
lateraloccipital_thickness	Occipital
lingual_thickness	Occipital
pericalcarine_thickness	Occipital
precuneus_thickness	Parietal
inferiorparietal_thickness	Parietal
paracentral_thickness	Frontal
postcentral_thickness	Parietal
superiorparietal_thickness	Parietal
supramarginal_thickness	Parietal
bankssts_thickness	Other
entorhinal_thickness	Temporal
fusiform_thickness	Temporal
inferiortemporal_thickness	Temporal
middletemporal_thickness	Temporal
superiortemporal_thickness	Temporal
temporalpole_thickness	Temporal
transversetemporal_thickness	Temporal
insula_thickness	Insula
"""


def load_lobes_map() -> pd.DataFrame:
    """

    References:
    - https://radiopaedia.org/articles/cingulate-gyrus
    - http://braininfo.rprc.washington.edu/centraldirectory.aspx?ID=159
    """
    with StringIO(_LOBE_MAPPING) as fin:
        lobes_map = pd.read_csv(fin, sep="\t", index_col=0)
    return lobes_map


# https://surfer.nmr.mgh.harvard.edu/fswiki/CMA
class UKBDataLoader:
    def __init__(self, csv_file: str, drop_outliers: bool = True) -> None:
        self._csv_file = Path(csv_file)
        self._drop_outliers = drop_outliers

    def load_freesurfer(self):
        # includes healthy subjects only
        all_features = pd.read_csv(self._csv_file, index_col=0)
        assert all_features.index.is_unique

        volumes, demographics = self._get_fs_vols_and_demographics(all_features)
        thickness = self._get_fs_thickness(all_features)

        thickness = thickness.reindex(index=volumes.index)
        if self._drop_outliers:
            volumes = drop_outliers(volumes)
            thickness = drop_outliers(thickness)
        volumes, thickness = volumes.align(thickness, axis=0, join="inner")

        return volumes, thickness, demographics.loc[volumes.index]

    def _get_fs_vols_and_demographics(self, all_features: pd.DataFrame):
        demo_cols = ["AGE", "SEX"]

        # https://www.mail-archive.com/freesurfer@nmr.mgh.harvard.edu/msg07933.html
        ventricle_cols = [
            'Left.Lateral.Ventricle',
            'Left.Inf.Lat.Vent',
            'Right.Lateral.Ventricle',
            'Right.Inf.Lat.Vent',
            'X3rd.Ventricle',
            'X4th.Ventricle',
            'X5th.Ventricle',
            'CSF',
            'Left.choroid.plexus',
            'Right.choroid.plexus',
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
            'Left.Cerebellum.Cortex',
            'Left.Thalamus.Proper',
            'Left.Caudate',
            'Left.Putamen',
            'Left.Pallidum',
            'Brain.Stem',
            'Left.Hippocampus',
            'Left.Amygdala',
            'Left.Accumbens.area',
            'Right.Cerebellum.Cortex',
            'Right.Thalamus.Proper',
            'Right.Caudate',
            'Right.Putamen',
            'Right.Pallidum',
            'Right.Hippocampus',
            'Right.Amygdala',
            'Right.Accumbens.area',
            'Optic.Chiasm',
            'eTIV',
        ]
        features = all_features.loc[:, volume_cols]

        # Sum up measurements of left and right hemisphere.
        column_pairs = {}
        for col in features.columns:
            for prefix in ("Left.", "Right.",):
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    column_pairs.setdefault(suffix, []).append(col)
                    break

        paired_data = {}
        for name, pair in column_pairs.items():
            assert len(pair) == 2
            paired_data[name] = features.loc[:, pair].sum(axis=1)
        paired_data = pd.DataFrame.from_dict(paired_data)

        drop_cols = [a for v in column_pairs.values() for a in v]
        features = pd.concat((paired_data, cc, ventricles, features.drop(drop_cols, axis=1)), axis=1)

        demographics = all_features.loc[:, demo_cols].reindex(index=features.index)

        return features, demographics

    def _get_fs_thickness(self, data: pd.DataFrame) -> pd.DataFrame:
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
        thick = {}
        for name in thickness_cols:
            thick[name] = data.loc[:, [f"lh_{name}", f"rh_{name}"]].mean(axis=1)

        stats = pd.DataFrame(thick, index=data.index)

        return stats


def get_lobes_map(thickness_measurements: pd.DataFrame) -> pd.DataFrame:
    lobes_map = load_lobes_map()
    lobes_map = lobes_map.loc[thickness_measurements.columns]
    lobes = lobes_map.Lobe.unique()
    colors = dict(zip(lobes, sns.color_palette("Dark2", n_colors=len(lobes))))
    lobes_map.loc[:, "color"] = lobes_map.Lobe.map(colors)

    return lobes_map


def prune_pairwise(cor_mat: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    idx = np.argmax(cor_mat.values)
    i, j = np.unravel_index(idx, shape=cor_mat.shape)
    val = cor_mat.iloc[i, j]
    if val > threshold:
        drop_var = cor_mat.index[j]
        pruned_mat = cor_mat.drop(index=drop_var, columns=drop_var)
        cor_mat = prune_pairwise(pruned_mat)
    return cor_mat


def _plot_corr(cor_mat: pd.DataFrame) -> None:
    if cor_mat.size == 1:
        LOG.warning("Skipping clustermap, because only one features is left after pruning")
    else:
        sns.clustermap(
            cor_mat, method="ward", metric="euclidean",
            row_cluster=False,
            square=True, annot=True, figsize=(7, 7), cmap="RdBu_r"
        )


def prune_by_group(data: pd.DataFrame, groups: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """For a pair of highly correlated features, keep one and drop the other."""
    pruned_cols = []
    for name, grp in groups.groupby("Lobe"):
        if len(grp.index) < 2:
            continue

        cor_mat_grp = data.loc[:, grp.index].corr(method="spearman")
        cor_mat_grp.values[np.diag_indices_from(cor_mat_grp)] = 0.0
        cor_mat_grp.rename_axis(name, inplace=True)

        cor_mat = prune_pairwise(cor_mat_grp)
        diff = cor_mat_grp.columns.difference(cor_mat.columns)
        LOG.info("%s: Pruned %d (%d -> %d)", name, len(diff), cor_mat_grp.shape[1], cor_mat.shape[1])
        pruned_cols.extend(diff)

        if plot:
            _plot_corr(cor_mat)

    return data.drop(pruned_cols, axis=1)


def _get_correlated_pair(cor_mat: pd.DataFrame, threshold: float = 0.7) -> pd.Index:
    idx = np.argmax(cor_mat.values)
    i, j = np.unravel_index(idx, shape=cor_mat.shape)
    val = cor_mat.iloc[i, j]
    new_cols = pd.Index([])
    if val > threshold:
        new_cols = new_cols.append(cor_mat.index[[i, j]])
    return new_cols


def _corr_without_diag(data, name):
    cor_mat_grp = data.corr(method="spearman")
    cor_mat_grp.values[np.diag_indices_from(cor_mat_grp)] = 0.0
    cor_mat_grp.rename_axis(name, inplace=True)
    return cor_mat_grp


def combine_by_group(data: pd.DataFrame, groups: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """Combine highly correlated features by creating a new feature that's the average."""
    pruned_data = []
    for name, grp in groups.groupby("Lobe"):
        if len(grp.index) < 2:
            continue

        data_grp = data.loc[:, grp.index]
        cor_mat_grp = _corr_without_diag(data_grp, name)

        while True:
            merge_cols = _get_correlated_pair(cor_mat_grp)
            if len(merge_cols) == 0:
                break
            merge_name = "+".join(merge_cols)
            merge_col = data_grp.loc[:, merge_cols].mean(axis=1).rename(merge_name)
            data_grp = pd.concat((data_grp.drop(merge_cols, axis=1), merge_col), axis=1)
            cor_mat_grp = _corr_without_diag(data_grp, name)

        diff = grp.index.difference(cor_mat_grp.columns)
        LOG.info("%s: Averaged %d (%d -> %d)", name, len(diff), len(grp.index), cor_mat_grp.shape[1])
        pruned_data.append(data_grp)

        if plot:
            _plot_corr(cor_mat_grp)

    return pd.concat(pruned_data, axis=1)


def apply_transform(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return apply_box_cox_and_standardize(data)


def get_volume_causes(confounders: pd.DataFrame) -> pd.DataFrame:
    dm = patsy.dmatrix(
        "AGE + I(AGE**2) + C(SEX) - 1",
        data=confounders, return_type="dataframe"
    )
    vol_causes = [
        "AGE",
        "I(AGE ** 2)",
        "C(SEX)[1]",
    ]
    dm = dm.loc[:, vol_causes]
    dm = (dm - dm.mean()) / dm.std(ddof=1)
    return dm
