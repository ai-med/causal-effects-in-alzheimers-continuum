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
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import patsy
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import special, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_random_state

from .pystan_models.model import PPCA, BPMF
from .utils import boxcox, drop_outliers, logt, stdt

LOG = logging.getLogger(__name__)

_FILE_UKB_VOLUMES = "data/ukb-data.csv"


def get_transforms(features, log_transform=None, bc_transform=None):
    if log_transform is None:
        log_transform = pd.Index([])
    if bc_transform is None:
        bc_transform = pd.Index([])

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


def load_data(standardize=False, remove_outliers=True):
    all_features = pd.read_csv(_FILE_UKB_VOLUMES, index_col=0)

    if "DX_GROUP" in all_features.columns:
        LOG.info("Including only healthy patients")
        all_features = all_features.loc[all_features.DX_GROUP == 0, :]
    demo_cols = ["AGE", "SEX"]

    meta_cols = pd.Index([
        "DX_GROUP",
        "MFS",
        "MANUFACTURER",
        "SCANNER_TYP",
        "ETHNIC",
        "SESSION_ID",
        "SESSION_TIME",
        "SET",
        "SITE_ID"
    ])

    cc_cols = [
        'CC_Posterior',
        'CC_Mid_Posterior',
        'CC_Central',
        'CC_Mid_Anterior',
        'CC_Anterior',
    ]
    cc = all_features.loc[:, cc_cols].sum(axis=1).rename("CC")

    features = all_features.drop(
        all_features.columns.intersection(meta_cols).tolist() + cc_cols,
        axis=1
    )

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

    paired_data.drop([
        "non.WM.hypointensities",
        "WM.hypointensities",
        "VentralDC",
    ], axis=1, inplace=True)

    other_features = [
        "CSF",
        "Optic.Chiasm",
        "X3rd.Ventricle",
        "X4th.Ventricle",
        "eTIV"
    ]

    features = pd.concat((paired_data, cc, all_features.loc[:, other_features]), axis=1)

    if remove_outliers:
        features = drop_outliers(features)
        has_zero = (features == 0.0).any(axis=1)
        LOG.info("Dropping %d samples due to zero volume", has_zero.sum())
        features.drop(has_zero.index[has_zero], inplace=True)
    demographics = all_features.loc[:, demo_cols].reindex(index=features.index)

    LOG.info("Final data size: %d %d", *features.shape)

    if standardize:
        sd = features.std(ddof=1)
        return (features - features.mean()) / sd

    return features, demographics, get_transforms(features, bc_transform=features.columns)


def get_volume_causes(dem: pd.DataFrame) -> pd.DataFrame:
    dm = patsy.dmatrix(
        "AGE + I(AGE**2) + C(SEX) - 1",
        data=dem, return_type="dataframe"
    )
    vol_causes = [
        "AGE",
        "I(AGE ** 2)",
        "C(SEX)[1]",
    ]
    dm = dm.loc[:, vol_causes]
    dm = (dm - dm.mean()) / dm.std(ddof=1)
    return dm


def get_ppca_deconfounder(
    data_volumes: pd.DataFrame,
    data_extra: pd.DataFrame,
    latent_dim: int,
    random_state: int,
    silent: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    assert len(data_volumes.index.symmetric_difference(data_extra.index)) == 0

    dm = get_volume_causes(data_extra)
    vol_causes = dm.columns.tolist()
    dm = pd.concat((dm, data_volumes), axis=1)

    m = PPCA(known_causes=vol_causes, latent_dim=latent_dim, random_state=random_state)
    m.fit(dm)

    pval = m.check_model(silent=silent)

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
    random_state: int,
    silent: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    dm = get_volume_causes(data_extra)
    vol_causes = dm.columns.tolist()
    dm = pd.concat((dm, data_volumes), axis=1)

    m = BPMF(vol_causes, latent_dim=latent_dim, random_state=random_state)
    m.fit(dm)

    pval = m.check_model(silent=silent)

    posterior = m.posterior_mean_estimates()

    # approximate the (random variable) substitute confounders with their inferred mean.
    Z_hat = pd.DataFrame(posterior.U, index=dm.index).add_prefix("Deconf_")

    recon = m.mean_reconstruction(posterior.U)
    resid = data_volumes.values - recon
    output = pd.DataFrame(resid, index=dm.index, columns=data_volumes.columns).add_suffix("-resid")
    return Z_hat, output, pval


def make_deconfounder(args):
    output_dir = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    out_file = output_dir / "augmented_data_{}_dim{}.pkl".format(args.model, args.latent_dim)
    if out_file.exists():
        raise RuntimeError(f"{out_file} already exists.")

    if args.model == "ppca":
        deconf_fn = get_ppca_deconfounder
    elif args.model == "bpmf":
        deconf_fn = get_bpmf_deconfounder
    else:
        assert False

    vols, dem, tfn = load_data(remove_outliers=True)

    data_vols_t = []
    for name, col in vols.iteritems():
        fn = tfn[name]
        col_t = pd.Series(fn(col.values), name=name, index=col.index)
        data_vols_t.append(col_t)
    data_vols_t = pd.concat((data_vols_t), axis=1)

    deconf, resid, pval = deconf_fn(data_vols_t, dem, latent_dim=args.latent_dim, random_state=2501)

    LOG.info("Posterior predictive check for %s (dim=%d): p-value = %.5f",
             args.model, args.latent_dim, pval)

    LOG.info("Writing %s", out_file.resolve())
    with open(out_file, "wb") as fout:
        pickle.dump((deconf, resid, pval), fout, protocol=pickle.HIGHEST_PROTOCOL)


class ConfoundingGenerator:

    def __init__(
        self,
        data: pd.DataFrame,
        known_causes: pd.DataFrame,
        sparsity: float = 0.8,
        prob_event: float = 0.5,
        var_x: float = 0.4,
        var_z: float = 0.4,
        num_sites: float = 3,
        random_state : int = None
    ) -> None:
        assert var_x + var_z < 1.0
        self.data = data
        self.known_causes = known_causes
        self.sparsity = sparsity
        self.prob_event = prob_event
        self.var_x = var_x
        self.var_z = var_z
        self.random_state = random_state
        self.num_sites = num_sites

        self._site_id = None

    def logistic(self, eta):
        probs = special.expit(eta)
        probs = np.maximum(1e-5, np.minimum(1.0 - 1e-5, probs))
        outcome = stats.bernoulli.rvs(p=probs, random_state=self._rnd)
        return outcome

    def get_site_assignments(self, data: pd.DataFrame) -> np.ndarray:
        if self._site_id is not None:
            return self._site_id

        p = PCA(n_components=2, random_state=self.random_state).fit(data.values)
        LOG.info("Explained variance: %s", p.explained_variance_ratio_)
        Xtc = np.dot(data.values, MinMaxScaler().fit_transform(p.components_.T))

        km = KMeans(
            n_clusters=self.num_sites,
            init="k-means++",
            n_init=10,
            max_iter=1000,
            tol=1e-6,
            algorithm="full",
            random_state=self.random_state
        ).fit(Xtc)
        site_id = km.predict(Xtc)  # cluster by all features
        self._site_id = site_id
        return site_id

    def plot_site_assignments(self, data: pd.DataFrame):
        _, ax = plt.subplots()
        p = PCA(n_components=2, random_state=self.random_state).fit(data.values)
        Xt = p.transform(data.values)
        Xtc = np.dot(data.values, MinMaxScaler().fit_transform(p.components_.T))
        var_perc = p.explained_variance_ratio_ * 100.

        km = KMeans(
            n_clusters=self.num_sites,
            init="k-means++",
            n_init=10,
            max_iter=1000,
            tol=1e-6,
            algorithm="full",
            random_state=self.random_state
        ).fit(Xtc)
        site_id = km.predict(Xtc)  # cluster by all features

        pal = sns.color_palette(n_colors=km.n_clusters)
        for i in range(km.n_clusters):
            m = site_id == i
            ax.scatter(Xt[m, 0], Xt[m, 1], color=pal[i], alpha=0.8)
        ax.set_xlabel("PC1 ({:.2f}%)".format(var_perc[0]))
        ax.set_ylabel("PC2 ({:.2f}%)".format(var_perc[1]))
        return ax

    def _make_sparse(self, x: np.ndarray) -> np.ndarray:
        """Only retain the largest coefficients"""
        o = np.abs(x).argsort()
        mask = o[:max(1, int(len(x) * self.sparsity))]
        x[mask] = 0.0
        return x

    def generate_outcome_with_site(self):
        self._rnd = check_random_state(self.random_state)
        var_noise = 1.0 - self.var_x - self.var_z
        assert var_noise > 0.0

        n_samples, n_features = self.data.shape

        # causal effect

        dem_data = get_volume_causes(self.known_causes)

        rv_x = stats.norm(loc=0.0, scale=0.5)
        coef_x = rv_x.rvs(size=n_features, random_state=self._rnd)
        coef_x = self._make_sparse(coef_x)

        causal_data = fit_regress_out(self.data, dem_data)

        mu_x = np.dot(causal_data.values, coef_x)
        contrib_x = np.sqrt(self.var_x) / np.std(mu_x, ddof=1)

        # confounding
        site_id = self.get_site_assignments(causal_data)

        # per group intercept and error variance
        intercepts = np.arange(1, self.num_sites + 1)
        err_var = stats.invgamma(a=3, loc=1).rvs(size=self.num_sites, random_state=self._rnd)
        LOG.info("Per-cluster intercepts: %s", intercepts)
        LOG.info("Per-cluster error variance: %s", err_var)

        mu_z_hidden = intercepts[site_id]
        contrib_z = np.sqrt(self.var_z * 0.9)  / np.std(mu_z_hidden, ddof=1)

        rv_age = stats.norm(loc=0.0, scale=0.2)
        coef_age = rv_age.rvs(random_state=self._rnd)
        mu_age = dem_data.loc[:, "AGE"].values * coef_age
        contrib_age = np.sqrt(self.var_z * 0.1) / np.std(mu_age, ddof=1)

        noise = stats.norm.rvs(loc=np.zeros(n_samples), scale=err_var[site_id], random_state=self._rnd)
        contrib_noise = np.sqrt(var_noise) / np.std(noise, ddof=1)

        mu = np.stack([mu_x, mu_z_hidden, mu_age, noise], axis=1)
        contrib = np.array([contrib_x, contrib_z, contrib_age, contrib_noise])

        # linear predictor
        eta = np.dot(mu, contrib)
        # center around 0
        eta -= np.mean(eta)

        intercept = stats.logistic.ppf(self.prob_event)
        eta += intercept

        true_coef = pd.Series(coef_x, index=self.data.columns)
        true_coef.loc["Intercept"] = intercept

        outcome = self.logistic(eta)

        conf_hidden = pd.Series(mu_z_hidden, index=self.data.index, name="unobserved_confounder")
        confounders = pd.concat((conf_hidden, dem_data.loc[:, "AGE"]), axis=1)
        return outcome, true_coef, confounders


def get_var_config(noise_var: float) -> Tuple[np.ndarray, np.ndarray]:
    assert noise_var > 0.
    assert noise_var < 1.

    r = np.array([0.666, 0.6, 0.4, 0.333, 0.25, 0.2, 0.1])
    r = np.concatenate(([1.0], r, np.reciprocal(r)))

    var_causal = r * (1.0 - noise_var) / (r + 1)
    var_confounded = (1.0 - noise_var) / (r + 1)
    return var_causal, var_confounded


def fit_model(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    est = LogisticRegression(penalty="none", C=1.0, solver="lbfgs", tol=1e-6, max_iter=10000)
    est.fit(X.values, y)
    coef = pd.Series(est.coef_.squeeze(), index=X.columns.str.replace(".resid", ""))
    coef["Intercept"] = float(est.intercept_)
    return coef


def fit_regress_out(X: pd.DataFrame, confounders: pd.DataFrame) -> pd.DataFrame:
    est = LinearRegression()
    X_resid = []
    for name, col in X.iteritems():
        est.fit(confounders.values, col.values)
        resid = pd.Series(col.values - est.predict(confounders.values), index=col.index, name=name)
        X_resid.append(resid)
    X_resid = pd.concat(X_resid, axis=1)

    return X_resid


def fit_and_get_coef(
    rep: int,
    gen: ConfoundingGenerator,
    data_ppca: pd.DataFrame,
    data_bpmf: pd.DataFrame,
    data_regress: pd.DataFrame,
    data_oracle: pd.DataFrame,
) -> Tuple[int, float, float, Dict[str, pd.Series]]:
    y, coef_actual, confounders = gen.generate_outcome_with_site()

    coefs_iter = {
        "Actual": coef_actual,
        "No_Deconf": fit_model(gen.data, y),
        "Regress_Out": fit_model(data_regress, y),
        "BPMF": fit_model(data_bpmf, y),
        "PPCA": fit_model(data_ppca, y),
        "Oracle": fit_model(pd.concat((data_oracle, confounders), axis=1), y)
    }

    return gen.random_state, gen.var_x, gen.var_z, coefs_iter


def estimate_coefs(args):
    vols, dem, tfn = load_data(remove_outliers=True)

    data_vols_t = []
    for name, col in vols.iteritems():
        fn = tfn[name]
        col_t = pd.Series(fn(col.values), name=name, index=col.index)
        data_vols_t.append(col_t)
    data_vols_t = pd.concat((data_vols_t), axis=1)

    with open(args.output_dir / "augmented_data_bpmf_dim{}.pkl".format(args.latent_dim), "rb") as fin:
        bpmf = pickle.load(fin)
        LOG.info("BPMF p-value: %f", bpmf[2])

    with open(args.output_dir / "augmented_data_ppca_dim{}.pkl".format(args.latent_dim), "rb") as fin:
        ppca = pickle.load(fin)
        LOG.info("PPCA p-value: %f", ppca[2])

    known_causes = get_volume_causes(dem)
    data_oracle = fit_regress_out(data_vols_t, known_causes)
    data_resid = fit_regress_out(data_vols_t, known_causes.loc[:, "AGE":"AGE"])

    var_causal, var_confounded = get_var_config(args.noise_var)

    random_state = 2501
    def iter_tasks():
        for var_x, var_z in zip(var_causal, var_confounded):
            for rep in range(args.n_repeat):
                gen = ConfoundingGenerator(
                    data=data_vols_t,
                    known_causes=dem,
                    var_x=var_x,
                    var_z=var_z,
                    num_sites=args.num_sites,
                    random_state=random_state + rep
                )
                yield rep, gen

    outputs = Parallel(n_jobs=args.n_jobs, verbose=2)(
        delayed(fit_and_get_coef)(i, gen, ppca[1], bpmf[1], data_resid, data_oracle)
        for i, gen in iter_tasks())

    out_file = args.output_dir / "coefs_dim{}.pkl".format(args.latent_dim)
    LOG.info("Writing %s", out_file.resolve())
    with open(out_file, "wb") as fout:
        pickle.dump(outputs, fout, protocol=pickle.HIGHEST_PROTOCOL)


def load_coefs(coefs_file: Path) -> pd.DataFrame:
    with open(coefs_file, "rb") as fin:
        coefs = pickle.load(fin)

    results = {}
    for _, var_x, var_z, coef in coefs:
        key = (var_x, var_z)
        results.setdefault(key, []).append(coef)

    df = {"overall_rmse": [], "method": [], "var_x": [], "var_z": [], "var_ratio": []}
    repeats = None
    for (var_x, var_z), coef in results.items():
        errors = {}
        n_repeats = len(coef)
        if repeats is None:
            repeats = n_repeats
        assert repeats == n_repeats, "expected {} experiments, but got {}".format(
            repeats, n_repeats)
        for cc in coef:
            actual = cc.pop("Actual").drop("Intercept")
            for method, other_cc in cc.items():
                err = (actual - other_cc.reindex_like(actual)).pow(2)
                if method in errors:
                    errors[method] += err
                else:
                    errors[method] = err

        for method, value in errors.items():
            rmse = np.sqrt((value / n_repeats).mean())
            df["overall_rmse"].append(rmse)
            df["method"].append(method)
            df["var_x"].append(var_x)
            df["var_z"].append(var_z)
            df["var_ratio"].append(var_x / var_z)

    df = pd.DataFrame.from_dict(df)
    return df


def evaluate_coefs(args):
    coefs_file = args.output_dir / "coefs_dim{}.pkl".format(args.latent_dim)
    df = load_coefs(coefs_file)

    ratios = np.array([(2, 3), (3, 5), (2, 5), (1, 3), (1, 4), (1, 5), (1, 10)])
    ratios = np.row_stack(([(1, 1)], ratios, ratios[:, (1, 0)]))
    ratios_float = ratios[:, 0] / ratios[:, 1]

    table = df.pivot(index="var_ratio", columns="method", values="overall_rmse")
    col_order = ["No_Deconf", "Regress_Out", "PPCA", "BPMF", "Oracle"]
    table = table.loc[:, col_order].sort_index(ascending=False)

    new_index = []
    for ratio in table.index:
        ratio_idx = np.argmin(np.abs(ratio - ratios_float))
        var_ratio = ratios[ratio_idx]
        new_index.append("{0:d}/{1:d}".format(*var_ratio))
    table.index = new_index

    delta = (-table.drop(["No_Deconf", "Oracle"], axis=1)
            .subtract(table.loc[:, "No_Deconf"], axis=0)
            .add_prefix("Improve "))
    table = pd.concat((table, delta), axis=1)

    out_file = args.output_dir / "evaluation_coefs_dim{}.csv".format(args.latent_dim)
    LOG.info("Writing %s", out_file.resolve())
    table.to_csv(out_file)


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    deconf_parser = subparsers.add_parser("deconf")
    deconf_parser.add_argument("--output_dir", type=Path, required=True)
    deconf_parser.add_argument("--model", choices=["bpmf", "ppca"], required=True)
    deconf_parser.add_argument("--latent_dim", type=int, default=2)
    deconf_parser.set_defaults(func=make_deconfounder)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--output_dir", type=Path, required=True)
    fit_parser.add_argument("--latent_dim", type=int, required=True)
    fit_parser.add_argument("--n_repeat", type=int, default=100)
    fit_parser.add_argument("--noise_var", type=float, default=0.1)
    fit_parser.add_argument("--num_sites", type=int, default=4)
    fit_parser.add_argument("--n_jobs", type=int, default=4)
    fit_parser.set_defaults(func=estimate_coefs)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--output_dir", type=Path, required=True)
    eval_parser.add_argument("--latent_dim", type=int, required=True)
    eval_parser.set_defaults(func=evaluate_coefs)

    args = parser.parse_args(args=args)

    logging.basicConfig(level=logging.INFO)
    np.seterr("raise")

    if not hasattr(args, "func"):
        parser.error("argument missing")

    args.func(args)


if __name__ == "__main__":
    main()
