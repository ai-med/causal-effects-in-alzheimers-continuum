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
from itertools import product
import logging
from pathlib import Path
import textwrap
from typing import Collection, Dict, Iterable, Optional

from matplotlib.patches import Rectangle
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import seaborn as sns

from ..ukb.data import load_lobes_map


def read_coef(filename: str, drop_intercept: bool = True) -> pd.DataFrame:
    drop_idx = ["mean_PPD", "log-posterior"]
    if drop_intercept:
        drop_idx.extend(["(phi)", "(Intercept)"])
    coef = pd.read_csv(filename, index_col=0).drop(drop_idx)
    coef = coef.stack().rename_axis(["coef", "variable"]).rename("value").reset_index()
    coef.loc[:, "is_thickness"] = coef.loc[:, "coef"].str.contains("_thickness")
    return coef


class FeatureRenamer:
    """https://www.slicer.org/wiki/Slicer3:Freesurfer_labels"""
    _FEATS_MAP = {
        'AGE': "Age",
        'PTGENDER': "Gender",
        'PTEDUCAT': "Education",
        'Lateral.Ventricle': 'Lateral ventricle',
        'Cerebellum.White.Matter': 'Cerebellum white matter',
        'Cerebellum.Cortex': 'Cerebellum cortex',
        'Thalamus.Proper': 'Thalamus proper',
        'Accumbens.area': 'Nucleus accumbens',
        'X3rd.Ventricle': '3rd Ventricle',
        'X4th.Ventricle': '4th Ventricle',
        'Brain.Stem': 'Brain Stem',
        'X5th.Ventricle': '5th Ventricle',
        'Optic.Chiasm': 'Optic chiasm',
        'CC': "Corpus callosum",
        'vessel': 'Perivascular space',
        'VentralDC': 'Ventral diencephalon',
        'Total_Ventricular_CSF': 'Total ventricular CSF',
    }
    # Thickness
    _FEATS_MAP_THICKNESS = {
        'bankssts_thickness': "Banks of superior temporal sulcus",
        'caudalanteriorcingulate_thickness': "Caudal anterior cingulate",
        'caudalmiddlefrontal_thickness': "Caudal middle",
        'cuneus_thickness': "Cuneus",
        'entorhinal_thickness': "Entorhinal",
        'frontalpole_thickness': "Frontal pole",
        'fusiform_thickness': "Fusiform",
        'inferiorparietal_thickness': "Inferior parietal lobule",
        'inferiortemporal_thickness': "Inferior temporal",
        'insula_thickness': "Insula",
        'isthmuscingulate_thickness': "Isthmus (cingulate gyrus)",
        'lateraloccipital_thickness': "Lateral occipital lobe",
        'lateralorbitofrontal_thickness': "Lateral orbitofrontal",
        'lingual_thickness': "Lingual Gyrus",
        'medialorbitofrontal_thickness': "Medial orbital gyrus",
        'middletemporal_thickness': "Middle temporal gyrus",
        'paracentral_thickness': "Paracentral lobule",
        'parahippocampal_thickness': "Parahippocampal",
        'parsopercularis_thickness': "Pars opercularis",
        'parsorbitalis_thickness': "Pars orbitalis",
        'parstriangularis_thickness': "Pars triangularis",
        'pericalcarine_thickness': "Peri calcarine sulcus",
        'postcentral_thickness': "Postcentral gyrus",
        'posteriorcingulate_thickness': "Posterior cingulate gyrus",
        'precentral_thickness': "Precentral gyrus",
        'precuneus_thickness': "Precuneus",
        'rostralanteriorcingulate_thickness': "Rostral anterior cingulate",
        'rostralmiddlefrontal_thickness': "Rostral middle frontal",
        'superiorfrontal_thickness': "Superior frontal gyrus",
        'superiorparietal_thickness': "Superior parietal lobule",
        'superiortemporal_thickness': "Superior temporal gyrus",
        'supramarginal_thickness': "Supramarginal gyrus",
        'temporalpole_thickness': "Temporal pole",
        'transversetemporal_thickness': "Transverse temporal gyrus",
    }

    def __init__(self, names: Collection[str], width: int = 17) -> None:
        feature_map = self._FEATS_MAP.copy()
        feature_map.update(self._FEATS_MAP_THICKNESS)
        for key, value in feature_map.items():
            feature_map[key] = textwrap.fill(value, width=width)

        self._feature_map = feature_map
        lobes_map = load_lobes_map().loc[:, "Lobe"]
        lobes_map.loc["bankssts_thickness"] = "Temporal"
        self._lobes_map = lobes_map.to_dict()
        self._rename_map = {}
        self._prepare(names, self._rename_map)

    def _prepare(self, names: Collection[str], rename_map: Dict[str, str]):
        aggregates = {}
        for name in names:
            key = name.replace(".resid", "")
            out_multiple = key.split("_thickness.")
            if len(out_multiple) == 1:
                rename_map[key] = self._feature_map.get(key, key)
            else:
                lobe = self._get_lobe_from_aggregate(out_multiple)
                aggregates.setdefault(lobe, {})[key] = out_multiple

        self._prepare_aggregates(aggregates, rename_map)

    def _get_lobe_from_aggregate(self, names: Collection[str]) -> str:
        lobes = set()
        for i, v in enumerate(names):
            if i == len(names) - 1:
                key = v
            else:
                key = f"{v}_thickness"
            lobes.add(self._lobes_map[key])
        assert len(lobes) == 1
        lobe = next(iter(lobes))
        return lobe

    def _prepare_aggregates(self, aggregates: Dict[str, Dict[str, Collection[str]]], rename_map: Dict[str, str]):
        for lobe, name_lists in aggregates.items():
            with_suffix = len(name_lists) > 1
            for i, (key, names) in enumerate(name_lists.items(), start=1):
                suffix = f" {'I' * i}" if with_suffix else ""
                rename_map[key] = f"{lobe} Lobe{suffix} ({len(names)} areas)"

    def rename(self, name: str) -> str:
        key = name.replace(".resid", "")
        return self._rename_map[key]


class CoefReader:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _iter_paths(self, dim: Optional[int] = None) -> Iterable[pd.DataFrame]:
        base_dir = self.base_dir
        yield read_coef(base_dir / "coef_adni_age_residualized.csv").assign(method="Regress-Out")
        yield read_coef(base_dir / "coef_adni_original.csv").assign(method="Non-Causal")

        yield read_coef(base_dir / "coef_adni_combat_residualized.csv").assign(
            method="ComBat"
        )

        if dim is None:
            for i, model in product(range(1, 9), ("bpmf", "ppca")):
                yield read_coef(base_dir / f"coef_adni_{model}_subst_conf_dim{i}.csv").assign(
                    method=f"{model} {i}"
                )
        else:
            yield read_coef(base_dir / f"coef_adni_bpmf_subst_conf_dim{dim}.csv").assign(
                method=f"BPMF"
            )
            yield read_coef(base_dir / f"coef_adni_ppca_subst_conf_dim{dim}.csv").assign(
                method=f"PPCA"
            )

    def read_coef_with_dim(self, dim: int = 6, text_wrap: int = 17) -> pd.DataFrame:
        data = pd.concat(list(self._iter_paths(dim)), ignore_index=True)

        renamer = FeatureRenamer(data.loc[:, "coef"].unique(), width=text_wrap)
        data.loc[:, "coef"] = data.loc[:, "coef"].map(renamer.rename)
        return data

    def read_all_coef(self) -> pd.DataFrame:
        all_data = list(self._iter_paths())

        all_data = pd.concat(all_data)
        renamer = FeatureRenamer(all_data.loc[:, "coef"].unique())
        all_data.loc[:, "coef"] = all_data.loc[:, "coef"].map(renamer.rename)
        return all_data


class Plotter:
    def __init__(self, coef_order, method_order, palette="Set1", figsize=(8, 15), width=0.8):
        self.coef_order = coef_order
        self.figsize = figsize
        self.method_order = method_order
        self.width = width
        self.linewidth = 1.3 * plt.rcParams["lines.linewidth"]
        self.markersize = .9 * plt.rcParams["lines.markersize"]
        self.dot_color = (0., 0., 0) #(178/255., 0, 29/255.)
        self.dot_color_alt = "darkgray"
        self.intervals = [("2.5%", "97.5%"), ("10%", "90%")]
#         self.orient = "vertical"
        self.orient = "horizontal"
        if isinstance(palette, str):
            palette = sns.color_palette(palette, n_colors=len(self.method_order))
        assert len(palette) == len(self.method_order)
        self.pal = dict(zip(
            self.method_order,
            palette
        ))

    def _plot_coef(self, ax, row, y, color):
        if self.orient == "vertical":
            lines_fn = ax.hlines
            dot_x = row.loc["mean"]
            dot_y = y
        else:
            lines_fn = ax.vlines
            dot_x = y
            dot_y = row.loc["mean"]

        is_significant = True
        for i, (start, end) in enumerate(self.intervals, start=1):
            signs = np.unique(np.sign(row.loc[[start, end]].values))
            is_significant = is_significant and len(signs) == 1
            lines_fn(y, row.loc[start], row.loc[end], linewidth=i * self.linewidth, color=color)

        dot_color = self.dot_color if is_significant else self.dot_color_alt
        ax.plot(dot_x, dot_y, "o", color=dot_color, markersize=self.markersize)

    def plot(self, data, legend_kwargs=None):
        _, ax = plt.subplots(figsize=self.figsize)

        n_methods = len(self.method_order)
        margin = 1 / n_methods * self.width
        width = (1 - 2 * margin)
        ticks = {}
        tick_text_above = {}
        areas = []
        for (coef_name, method), df in data.groupby(["coef", "method"]):
            method_idx = self.method_order.index(method)
            coef_idx = len(self.coef_order) - self.coef_order.index(coef_name) - 1
            row = df.pivot(index="variable", columns="coef", values="value")

            offset = method_idx / (n_methods - 1) * width - 0.5 + margin
            ticks[coef_idx] = coef_name
            tick_text_above[coef_idx] = "T" if df.loc[:, "is_thickness"].iloc[0] else "V"
            y = coef_idx + offset
            areas.append(y)
            self._plot_coef(ax, row, y, self.pal[method])

        if self.orient == "vertical":
            self._decorate_v(ax, ticks)
        else:
            self._decorate_h(ax, ticks, tick_text_above)
        self.legend(ax, legend_kwargs)

        return ax

    def _decorate_v(self, ax, ticks):
        ax.set_yticks(np.arange(len(self.coef_order)))
        ax.set_yticklabels(ticks[i] for i in range(len(self.coef_order)))
        ax.xaxis.grid(True)
        ax.set_ylim(-.5, len(self.coef_order) - 0.5)
        ax.axvline(0.0, linestyle="dashed", color="gray")

    def _decorate_h(self, ax, ticks, tick_text_above):
        ax.set_xticks(np.arange(len(self.coef_order)))
        ax.set_xticklabels(ticks[i] for i in range(len(self.coef_order)))
        ax.yaxis.grid(True)
        ax.set_xlim(-.5, len(self.coef_order) - 0.5)
        ax.axhline(0.0, linestyle="dashed", color="gray")
        ax.set_ylabel("Coefficient")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        a = np.concatenate((
            [-0.5],
            np.repeat(np.arange(0.5, len(ticks), 1), 2),
            [len(ticks) + 0.5]
        ))
        b = np.repeat(np.arange(len(ticks)) % 2 != 0, 2)

        ymin, ymax = ax.get_ylim()
        collection = mc.BrokenBarHCollection.span_where(
            a, ymin=ymin, ymax=ymax, where=b > 0, facecolor='#deebf7', alpha=0.4)
        ax.add_collection(collection)

        bottom_y = 0.015
        for x, t in tick_text_above.items():
            plt.text(x, bottom_y, t, ha="center", transform=ax.get_xaxis_transform())

    def legend(self, ax, legend_kwargs):
        handles = []
        labels = []
        for name, color in self.pal.items():
            p = Rectangle((0, 0), 1, 1)
            p.set_facecolor(color)
            handles.append(p)
            labels.append(name)

        if legend_kwargs is None:
            legend_kwargs = {"loc": "best"}

        ax.legend(handles, labels, **legend_kwargs)


def plot_coefs(args):
    output_dir = args.outputs_dir
    parameters = {
        "base_dir": str(args.models_dir.resolve()),
    }

    pm.execute_notebook(
        Path(__file__).parent / "notebooks" / "plot-betareg-coef.ipynb",
        output_dir / "plot-betareg-coef_outputs.ipynb",
        parameters=parameters,
        progress_bar=True,
    )


def plot_ace(args):
    output_dir = args.outputs_dir
    parameters = {
        "data_path": str(args.data_file.resolve()),
        "subst_conf_dir": str(args.subst_conf_dir.resolve()),
        "models_dir": str(args.models_dir.resolve()),
        "n_jobs": args.n_jobs,
    }

    pm.execute_notebook(
        Path(__file__).parent / "notebooks" / "estimate_ace.ipynb",
        output_dir / "estimate_ace_outputs.ipynb",
        parameters=parameters,
        progress_bar=True,
    )


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    p = subparsers.add_parser("coefs")
    p.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Path to directory containing CSV files of fitted models.",
    )
    p.add_argument(
        "--outputs_dir",
        type=Path,
        required=True,
        help="Path to directory where to write outputs to.",
    )
    p.set_defaults(func=plot_coefs)

    p = subparsers.add_parser("ace")
    p.add_argument(
        "--data_file",
        type=Path,
        required=True,
        help="Path to HDF5 file containing volume and thickness measurements."
    )
    p.add_argument(
        "--models_dir",
        type=Path,
        required=True,
        help="Path to directory containing CSV files of fitted models.",
    )
    p.add_argument(
        "--subst_conf_dir",
        type=Path,
        required=True,
        help="Path to directory with HDF5 files of containing estimated substitute confounders."
    )
    p.add_argument(
        "--outputs_dir",
        type=Path,
        required=True,
        help="Path to directory where to write outputs to.",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=5,
        help="The maximum number of concurrently running jobs. Default: %(default)s",
    )
    p.set_defaults(func=plot_ace)

    args = parser.parse_args(args=args)
    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
