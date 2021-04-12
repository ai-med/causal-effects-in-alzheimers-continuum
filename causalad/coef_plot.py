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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import seaborn as sns
from matplotlib.patches import Rectangle

feats_map = {
    'AGE': "Age",
    'PTGENDER': "Gender",
    'PTEDUCAT': "Education",
    'Lateral.Ventricle': 'Lateral Ventricle',
    'Cerebellum.White.Matter': 'Cerebellum\nWhite Matter',
    'Cerebellum.Cortex': 'Cerebellum Cortex',
    'Thalamus.Proper': 'Thalamus Proper',
    'Accumbens.area': 'Accumbens area',
    'X3rd.Ventricle': '3rd Ventricle',
    'X4th.Ventricle': '4th Ventricle',
    'Brain.Stem': 'Brain Stem',
    'X5th.Ventricle': '5th Ventricle',
    'Optic.Chiasm': 'Optic Chiasm',
    'CC': "Corpus Callosum",
    'vessel': "Perivascular Space",
    "VentralDC": "Ventral Diencephalon",
}
# Thickness
feats_map_thickness = {
    'parahippocampal_thickness': "Parahippocampal ",
    'caudalanteriorcingulate_thickness': "Caudal Anterior\nCingulate",
    'fusiform_thickness': "Fusiform",
    'lateralorbitofrontal_thickness': "Lateral\nOrbitofrontal",
    'insula_thickness': "Insular",
    'entorhinal_thickness': "Entorhinal",
    'precuneus_thickness': "Precuneus",
    'rostralmiddlefrontal_thickness': "Rostral\nMiddle Frontal",
}
feats_map.update(feats_map_thickness)


def rename_coef(val: str) -> str:
    x = val.replace(".resid", "")
    x = feats_map.get(x, x)
    return x


class Plotter:
    def __init__(self, coef_order, method_order, palette="Set1", figsize=(8, 15)):
        self.coef_order = coef_order
        self.figsize = figsize
        self.method_order = method_order
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

    def plot(self, data):
        _, ax = plt.subplots(figsize=self.figsize)

        n_methods = len(self.method_order)
        margin = 1 / n_methods * 0.8
        width = (1 - 2 * margin)
        ticks = {}
        areas = []
        for (coef_name, method), df in data.groupby(["coef", "method"]):
            method_idx = self.method_order.index(method)
            coef_idx = len(self.coef_order) - self.coef_order.index(coef_name) - 1
            row = df.pivot(index="variable", columns="coef", values="value")

            offset = method_idx / (n_methods - 1) * width - 0.5 + margin
            ticks[coef_idx] = coef_name
            y = coef_idx + offset
            areas.append(y)
            self._plot_coef(ax, row, y, self.pal[method])

        if self.orient == "vertical":
            self._decorate_v(ax, ticks)
        else:
            self._decorate_h(ax, ticks)
        self.legend(ax)

        return ax

    def _decorate_v(self, ax, ticks):
        ax.set_yticks(np.arange(len(self.coef_order)))
        ax.set_yticklabels(ticks[i] for i in range(len(self.coef_order)))
        ax.xaxis.grid(True)
        ax.set_ylim(-.5, len(self.coef_order) - 0.5)
        ax.axvline(0.0, linestyle="dashed", color="gray")

    def _decorate_h(self, ax, ticks):
        ax.set_xticks(np.arange(len(self.coef_order)))
        ax.set_xticklabels(ticks[i] for i in range(len(self.coef_order)))
        ax.yaxis.grid(True)
        ax.set_xlim(-.5, len(self.coef_order) - 0.5)
        ax.axhline(0.0, linestyle="dashed", color="gray")

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
        for x, name in ticks.items():
            if name in feats_map_thickness.values():
                t = "T"
            else:
                t = "V"
            plt.text(x, bottom_y, t, ha="center", transform=ax.get_xaxis_transform())

    def legend(self, ax):
        handles = []
        labels = []
        for name, color in self.pal.items():
            p = Rectangle((0, 0), 1, 1)
            p.set_facecolor(color)
            handles.append(p)
            labels.append(name)

        ax.legend(handles, labels, loc="best")
