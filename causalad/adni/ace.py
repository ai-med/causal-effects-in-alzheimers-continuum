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
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy import stats

from .plot import FeatureRenamer
from ..ukb.io import load_deconfounder_residuals


def to_r_name(x):
    return x.replace("-", ".").replace("+", ".")


class BetaRegPredictor:
    def __init__(self, coef_samples_path: str) -> None:
        samples = pd.read_csv(coef_samples_path)
        samples.columns = samples.columns.str.replace(r"\.resid$", "", regex=True)
        self.coef_samples = samples  # shape = (n_draws, n_features)
        self.intercept_samples = samples.loc[:, "(Intercept)"].values  # shape = (n_draws,)
        self.phi_samples = samples.loc[:, "(phi)"].values  # shape = (n_draws,)

    def predict(self, data_in: pd.DataFrame, random_state: int = 1365) -> np.ndarray:
        data_in = data_in.rename(columns=to_r_name)
        data, coefs = data_in.align(self.coef_samples, axis=1, join="inner")
        assert data.shape == data_in.shape, data.columns.symmetric_difference(data_in.columns)
        lp = np.dot(data.values, coefs.values.T) + self.intercept_samples

        mu = expit(lp)  # shape = (n_samples, n_draws)

        yhat = stats.beta(
            a=mu * self.phi_samples,
            b=(1 - mu) * self.phi_samples
        ).rvs(random_state=random_state)
        return yhat


class FeatureTransformer:
    def __init__(self, data_path: str, residuals_path: Optional[str]) -> None:
        self.data_path = data_path
        self.n_samples = 100

        self._read_features_and_transforms()
        if residuals_path is None:
            self.predicted = pd.DataFrame(0.0, index=self.features.index, columns=self.features.columns)
        else:
            if residuals_path.endswith(".h5"):
                resid = load_deconfounder_residuals(residuals_path)
            else:
                resid = pd.read_csv(residuals_path, index_col=0).drop("ADAS13", axis=1)

            # get predicted value from BPFM: residuals = actual - predicted
            self.predicted = self.features_t.add_suffix("-resid") - resid
            self.predicted.columns = self.features.columns

    def _read_features_and_transforms(self):
        with pd.HDFStore(self.data_path, mode="r") as store:
            tiv = store.get("tiv").loc[:, "eTIV"]
            transforms = store.get("transforms")
            volumes_t = store.get("volumes")
            thickness_t = store.get("thickness")

        # invert transforms for original data
        features_t = pd.concat((volumes_t, thickness_t), axis=1)

        self.is_thickness = pd.Series(False, index=features_t.columns)
        self.is_thickness.loc[thickness_t.columns] = True

        features = {}
        for name, feat_t in features_t.iteritems():
            t = transforms.loc[name]
            feat_inv = self.inverse_transform(feat_t.values, t)
            if not self.is_thickness[name]:
                feat_inv *= tiv.values  # volumes are divided by TIV
            assert np.isfinite(feat_inv).all(), name
            features[name] = feat_inv

        self.features = pd.DataFrame.from_dict(features)
        self.features.index = features_t.index
        self.features_t = features_t
        self.tiv = tiv
        self.transforms = transforms

    def inverse_transform(self, values: np.ndarray, t: pd.Series) -> np.ndarray:
        val_inv = np.power(
            (values * t.loc["stddev"] + t.loc["mean"]) * t.loc["lmbda"] + 1,
            1.0 / t.loc["lmbda"]
        )
        return val_inv

    def transform(self, values: np.ndarray, t: pd.Series) -> np.ndarray:
        val_t = stats.boxcox(values, lmbda=t.loc["lmbda"])
        val_t -= t.loc["mean"]
        val_t /= t.loc["stddev"]
        return val_t

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data_t = pd.DataFrame(index=data.index, columns=data.columns)
        tiv = self.tiv.loc[data.index]
        pred = self.predicted.loc[data.index]
        for feature_name, series in data.iteritems():
            val_t = series.values.copy()
            if not self.is_thickness[feature_name]:
                val_t /= tiv  # divide by TIV

            # feature on transformed scale
            transf = self.transforms.loc[feature_name]
            val_t = self.transform(val_t, transf)
            assert np.isfinite(val_t).all()

            # residuals of feature
            v = val_t - pred.loc[:, feature_name].values
            data_t.loc[:, feature_name] = v
        return data_t

    def get_samples(self, feature_name: str, subset: Sequence[Any]):
        range_org = self.features.loc[:, feature_name].agg([np.min, np.max])

        # feature on the original scale
        samples = np.linspace(range_org.loc["amin"], range_org.loc["amax"], self.n_samples)

        div_tiv = not self.is_thickness[feature_name]
        transf = self.transforms.loc[feature_name]

        tiv = self.tiv.loc[subset].values
        pred = self.predicted.loc[subset, feature_name].values
        features_intervened = self.features_t.loc[subset].copy()
        val = np.empty(len(subset), dtype=float)
        for val_org in samples:
            val[:] = val_org
            if div_tiv:
                val /= tiv  # divide by TIV

            # feature on transformed scale
            val_t = self.transform(val, transf)
            assert np.isfinite(val_t).all()

            # residuals of feature
            v = val_t - pred
            features_intervened.loc[:, feature_name] = v
            yield val_org, features_intervened


class AceEstimator:
    def __init__(self, feat_transformer: FeatureTransformer, coef_samples_path: str) -> None:
        self.betareg = BetaRegPredictor(coef_samples_path)
        self.feat_transformer = feat_transformer

    def compute_ace(self, fname: str, subset: Sequence[Any]) -> Tuple[str, Tuple[np.ndarray, np.ndarray]]:
        ft = self.feat_transformer
        px = np.empty(ft.n_samples, dtype=float)
        py = np.empty(ft.n_samples, dtype=float)
        for i, (intervention, data_t) in enumerate(ft.get_samples(fname, subset)):
            yhat = self.betareg.predict(data_t)
            yhat_mean = yhat.mean()  # average over posterior draws and samples in the data
            px[i] = intervention
            py[i] = yhat_mean
        return fname, (px, py)


class AxesGridIter:
    def __init__(self, shape: Tuple[int, int]) -> None:
        self.n_cols = shape[1]
        self.total = (shape[0] // 2) * shape[1]

    def __iter__(self) -> Iterable[Tuple[int, int]]:
        row_idx = 0
        col_idx = 0
        for _ in range(self.total):
            yield row_idx, col_idx
            if col_idx + 1 == self.n_cols:
                row_idx += 2
                col_idx = 0
            else:
                col_idx += 1


class Plotter:
    def __init__(
        self,
        features: Sequence[str],
        col_order: Optional[Sequence[str]] = None,
        wrap_cols: Optional[int] = None,
        max_adas: int = 85,
        legend_out: bool = False,
    ) -> None:
        self.features = features
        self.col_order = col_order
        self.wrap_cols = wrap_cols
        self.max_adas = max_adas
        self.legend_out = legend_out

    def _plot_ace_line(self, px, py, ax, title, **kwargs):
        ax.plot(px, py * 85, **kwargs)
        ax.set_ylim(0, self.max_adas)
        ax.grid(True)
        ax.set_title(title)

    def _plot_feature_dist(self, fname, ax):
        ax.boxplot(
            self.features.loc[:, fname],
            vert=False,
            widths=0.75,
            showmeans=True,
            medianprops={"color": "#ff7f00"},
            meanprops={"marker": "d", "markeredgecolor": "#a65628", "markerfacecolor": "none"},
        )
        ax.yaxis.set_visible(False)
        #ax.hist(ft.features.loc[:, fname].values, bins="auto", density=True)
        ax.set_xlabel("$x$")

    def _make_figure(self, n_features: int):
        n_cols = self.wrap_cols or n_features
        n_rows = int(np.ceil(n_features / n_cols))
        h_ratios = [5 * (self.max_adas / 85), 1]

        fig = plt.figure(figsize=(3.6 * n_cols, 6 * n_rows * (self.max_adas / 85)))
        gs_outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.2 * np.ceil(85 / self.max_adas))

        axs = np.empty((2 * n_rows, n_cols), dtype=object)
        for i, j in product(range(n_rows), range(n_cols)):
            gs = gs_outer[i, j].subgridspec(2, 1, height_ratios=h_ratios, hspace=0.1 * (85 / self.max_adas))

            ax_top = fig.add_subplot(gs[0, 0])
            ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
            ax_top.tick_params(axis="x", labelbottom=False)
            row_idx = i * 2
            axs[row_idx, j] = ax_top
            axs[row_idx + 1, j] = ax_bot

        return fig, axs

    def _get_renamer(self):
        return FeatureRenamer(self.features.columns.map(to_r_name), width=30)

    def _label_axes(self, axs):
        for ax in axs[::2, 0]:
            ax.set_ylabel(r"$\mathbb{E}(\mathrm{ADAS}\,|\,do(x))$")

        for ax in axs[1::2, 0]:
            ax.yaxis.set_visible(True)
            ax.set_yticklabels(["Observed"])

    def plot_ace(self, causal_effects: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        renamer = self._get_renamer()

        n_features = len(causal_effects)
        fig, axs = self._make_figure(n_features)
        col_order = self.col_order or causal_effects.keys()

        axs_iter = AxesGridIter(axs.shape)
        for (row_idx, col_idx), fname in zip(axs_iter, col_order):
            px, py = causal_effects[fname]
            title = renamer.rename(to_r_name(fname))

            self._plot_ace_line(px, py, axs[row_idx, col_idx], title)

            self._plot_feature_dist(fname, axs[row_idx + 1, col_idx])

        self._label_axes(axs)

        return fig

    def compare_ace(
        self,
        causal_effects: Sequence[Dict[str, Tuple[np.ndarray, np.ndarray]]],
        names: Sequence[str],
        styles: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        assert len(causal_effects) == len(names)
        keys = self.col_order or frozenset(causal_effects[0].keys())

        styles = styles or {}

        renamer = self._get_renamer()
        n_features = len(keys)
        fig, axs = self._make_figure(n_features)

        axs_iter = AxesGridIter(axs.shape)
        for fname, (row_idx, col_idx) in zip(keys, axs_iter):
            title = renamer.rename(to_r_name(fname))
            effect_iter = map(lambda x: x[fname], causal_effects)
            for label, (px, py) in zip(names, effect_iter):
                style = styles.get(label, {})
                self._plot_ace_line(
                    px, py, axs[row_idx, col_idx], title, label=label, **style,
                )

            self._plot_feature_dist(fname, axs[row_idx + 1, col_idx])

        self._label_axes(axs)
        if self.legend_out:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            axs[-2, (n_features % self.wrap_cols) - 1].legend(
                handles, labels, loc="center left", bbox_to_anchor=(1.1, 0.5)
            )
        else:
            axs[0, 0].legend(loc="best")
        # hide unused subplots
        for ax in axs[row_idx:, (col_idx + 1):].flat:
            ax.set_visible(False)

        return fig


def composite_ace(
    features: pd.DataFrame,
    transform_func: Callable[[pd.DataFrame], pd.DataFrame],
    predictors: Dict[str, BetaRegPredictor],
    q_low: int = 25,
    q_high: int = 50,
) -> pd.DataFrame:
    q_low_f = q_low / 100.
    q_high_f = q_high / 100.
    q = features.quantile([q_low_f, q_high_f])

    def transform_data(data: np.ndarray):
        data_in = pd.DataFrame(
            np.tile(data, (features.shape[0], 1)),
            index=features.index,
            columns=features.columns,
        )
        data_t = transform_func(data_in)
        return data_t

    data_t_low = transform_data(q.loc[q_low_f].values)
    data_t_high = transform_data(q.loc[q_high_f].values)

    idx = pd.MultiIndex.from_product([(q_low, q_high), ("mean", "std")])
    combined_effect = pd.DataFrame(index=list(predictors.keys()), columns=idx, dtype=float)
    stddev = partial(np.std, ddof=1)
    for model, pred in predictors.items():
        ace_samples = pred.predict(data_t_low).mean(axis=0)
        combined_effect.loc[model, q_low] = np.mean(ace_samples), stddev(ace_samples)
        ace_samples = pred.predict(data_t_high).mean(axis=0)
        combined_effect.loc[model, q_high] = np.mean(ace_samples), stddev(ace_samples)

    return combined_effect * 85.0


def plot_composite_ace(
    data: pd.DataFrame,
    palette: Sequence[Tuple[float, float, float]],
    marker: str = "o",
    figsize: Tuple[float, float] = (4, 5),
):
    if data.shape[1] != 2:
        raise ValueError(f"expected data with 2 columns, but got {data.shape[1]}")

    _, ax = plt.subplots(figsize=figsize)

    for j, (_, col) in enumerate(data.iteritems()):
        for i, (name, value) in enumerate(col.iteritems()):
            fc = "none" if j == 0 else palette[i]
            ax.plot(
                i, value, marker=marker, markeredgecolor=palette[i], markerfacecolor=fc, label=name
            )

    for i, row in enumerate(data.itertuples(index=False)):
        dy = row[0] - row[1]
        ax.plot([i, i], [row[1] + 1, row[0] - 1.3], color=palette[i], linestyle="--")

        head_width = 0.25
        p = Polygon(
            [
                [i - head_width / 2, row[0] - 1.2],
                [i + head_width / 2, row[0] - 1.2],
                [i, row[0] - 1.2 + 1.75 * head_width]
            ],
            color=palette[i],
        )
        ax.add_patch(p)

        ax.text(i + 0.1, row[1] + dy / 2, f"{dy:+.1f}", va="center", ha="left")

    ax.set_ylabel(r"$\mathbb{E}[\mathrm{ADAS}\,|\, do(x)]$")
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.index.tolist(), rotation=45, rotation_mode="anchor", ha="right")
    ax.yaxis.grid(True)
    ax.set_xlim(-1, data.shape[0])

    handles = [
        Line2D([0], [0], marker=marker, markeredgecolor="black", markerfacecolor=mf, linestyle="")
        for mf in ["none", "black"]
    ]

    ax.legend(
        handles,
        ["$x_{{ {:d} }}$".format(v) for v in data.columns],
        fontsize="large",
        handlelength=0,
        borderpad=0.8,
    )
    return ax


def barplot_composite_ace(
    data,
    colors=["#BF6C78", "#97CAEB"],
    width=0.35,
    figsize=(8, 5),
):
    _, ax = plt.subplots(figsize=figsize)

    ace_mean = data.xs("mean", level=1, axis=1)
    ace_stddev = data.xs("std", level=1, axis=1)
    x = np.arange(ace_mean.shape[0])
    labels = ["$x_{{ {:d} }}$".format(v) for v in ace_mean.columns]

    offsets = [-width / 2, width / 2]
    for (_, values), (_, err), label, color, offset in zip(
        ace_mean.iteritems(),
        ace_stddev.iteritems(),
        labels,
        colors,
        offsets,
    ):
        ax.bar(x + offset, values, width, yerr=err, label=label, color=color, zorder=3)

    ypad = 2
    barheight = 1
    for i, (_, val) in enumerate(ace_mean.iloc[:, 0].iteritems()):
        y = val + ace_stddev.iloc[i, 0]
        ax.plot(
            np.array([x[i] + offsets[0], x[i] + offsets[0], x[i] + offsets[1], x[i] + offsets[1]]),
            np.array([y - barheight, y, y, y - barheight]) + ypad,
            '-',
            color="gray"
        )

        diff = abs(val - ace_mean.iloc[i, 1])
        ax.text(x[i], y + ypad + 1, f"$\Delta = {diff:.1f}$", ha="center")

    ax.set_ylabel(r"$\mathbb{E}[\mathrm{ADAS}\,|\, do(x)]$")
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_xticklabels(data.index.tolist())
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", fontsize="large")
    ax.set_ylim(0, 50)

    return ax
