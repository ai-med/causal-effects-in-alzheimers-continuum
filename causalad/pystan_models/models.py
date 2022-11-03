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
import importlib
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from collections import namedtuple
from tempfile import NamedTemporaryFile
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pystan.misc
from joblib import Parallel, delayed
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state
from tqdm import tqdm

RandomStateType = Union[int, np.random.RandomState, None]

InferredPPCAParameters = namedtuple(
    "InferredPPCAParameters",
    ("W_mu", "Z_mu", "sigma_x"),
)


def make_holdout_mask(n_samples: int,
                      n_features: int,
                      holdout_portion: float = 0.2,
                      random_state: RandomStateType = None) -> Tuple[np.ndarray, np.ndarray]:
    """randomly holdout some entries of X"""
    rnd = check_random_state(random_state)
    holdout_mask = rnd.binomial(1, p=holdout_portion, size=(n_samples, n_features))
    holdout_row, _ = holdout_mask.nonzero()

    holdout_subjects = np.unique(holdout_row)
    return holdout_mask, holdout_subjects


class Model(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        posterior_samples: int,
        stan_class: type,
        random_state: RandomStateType = None,
    ) -> None:
        self.posterior_samples = posterior_samples
        self.random_state = random_state
        self._stan_class = stan_class

    @abstractmethod
    def _get_data(self, X, y):
        """Return data for StanModel.sampling"""

    def _fit(self, stan_data):
        seed = pystan.misc._check_seed(self.random_state)
        fit = self._stan_class(stan_data, seed)

        pars = fit._get_param_names()

        f = NamedTemporaryFile(suffix='output.csv', delete=False)

        stan_args = {
            "iter": 20000,
            "init": "random",
            "chain_id": 1,
            "seed": seed,
            "method": "variational",
            "algorithm": "meanfield",
            "tol_rel_obj": 0.00001,
            "output_samples": self.posterior_samples,
            "sample_file": f.name,
        }
        stan_args = pystan.misc._get_valid_stan_args(stan_args)

        ret, sample = fit._call_sampler(stan_args, pars_oi=pars)

        self._results = {
            "sampler_params": sample.sampler_params,
            "sampler_param_names": sample.sampler_param_names,
            "mean_pars": sample.mean_pars,
            "mean_par_names": sample.mean_par_names,
        }
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> object:
        """Fit model"""
        stan_data = self._get_data(X, y)
        return self._fit(stan_data)

    def get_samples(self, name: str, is_array: bool = True) -> np.ndarray:
        """Retrieve posterior samples for given variable"""
        params = self._results["sampler_params"]
        names = self._results["sampler_param_names"]
        if is_array:
            name += "["
        values = [i for i, v in zip(params, names) if v.startswith(name)]
        return np.array(values)

    def get_mean(self, name: str, is_array: bool = True) -> np.ndarray:
        """Retrieve mean for given variable"""
        params = self._results["mean_pars"]
        names = self._results["mean_par_names"]
        if is_array:
            name += "["
        values = [i for i, v in zip(params, names) if v.startswith(name)]
        return np.array(values)


class PPCA(Model):
    def __init__(
        self,
        latent_dim: int,
        sigma_w: float = 2.0,
        standardize: bool = True,
        posterior_samples: int = 1000,
        random_state: RandomStateType = None,
    ) -> None:
        from ._ppca4 import StanFit4Model

        super().__init__(
            posterior_samples=posterior_samples,
            stan_class=StanFit4Model,
            random_state=random_state,
        )
        self.latent_dim = latent_dim
        self.sigma_w = sigma_w
        self.standardize = standardize

    def _get_data(self, X, y):
        """Return data for StanModel.sampling"""
        if self.standardize:
            Xt = StandardScaler().fit_transform(X)
        else:
            Xt = check_array(X, copy=True)

        self.holdout_mask_, self.holdout_subjects_ = make_holdout_mask(
            n_samples=X.shape[0],
            n_features=X.shape[1],
            random_state=self.random_state)
        self.n_samples_, self.n_features_ = X.shape
        self.X_valid_ = Xt * self.holdout_mask_

        stan_data = {
            'N': Xt.shape[0],
            'D': Xt.shape[1],
            'DZ': self.latent_dim,
            'X': Xt,
            'sigma_w': self.sigma_w,
            'holdout': self.holdout_mask_,
        }
        return stan_data

    def replicated_data(self) -> np.ndarray:
        """Retrieve data generated from the predictive distribution

        Returns
        -------
        X_rep : np.ndarray, shape = (n_draws, n_samples, n_features)
        """
        X_rep = self._reconstruction_samples()
        # look only at the heldout entries
        holdout_gen = np.multiply(X_rep, self.holdout_mask_[np.newaxis])
        return holdout_gen

    def _reconstruction_mean(self) -> np.ndarray:
        W = self.get_samples("W")
        # make first dimension the number of posterior samples
        W = np.stack(
            [W[:, i].reshape((self.n_features_, self.latent_dim), order="F")
            for i in range(W.shape[1])]
        )

        Z = self.get_samples("Z")
        # make first dimension the number of posterior samples
        Z = np.stack(
            [Z[:, i].reshape((self.latent_dim, self.n_samples_), order="F")
            for i in range(Z.shape[1])]
        )

        # Z.T @ W.T
        mu = np.einsum("aji,akj->aik", Z, W)
        return mu

    def _reconstruction_std(self) -> np.ndarray:
        sigma_x = self.get_samples("sigma_x", False).squeeze(0)
        return sigma_x[:, np.newaxis, np.newaxis]

    def _reconstruction_samples(self) -> np.ndarray:
        mu = self._reconstruction_mean()
        sigma_x = self._reconstruction_std()
        samples = stats.norm(loc=mu, scale=sigma_x).rvs()
        return samples

    def posterior_mean_estimates(self) -> InferredPPCAParameters:
        """Retrieve posterior mean estimates."""
        latent_dim = self.latent_dim
        w_mean = self.get_mean("W").reshape((-1, latent_dim), order="F")
        z_mean = self.get_mean("Z").reshape((latent_dim, -1), order="F").T
        sigma_x_mean = self.get_mean("sigma_x", False)

        return InferredPPCAParameters(w_mean, z_mean, sigma_x_mean)

    def check_model(self, n_eval: int = 100, silent: bool = False) -> float:
        """Posterior predictive check aka Bayesian p-value"""
        x_valid = self.X_valid_

        holdout_gen = self.replicated_data()
        est = self.posterior_mean_estimates()

        w_rv = stats.norm(loc=est.W_mu, scale=self.sigma_w)
        z_rv = [stats.norm(loc=z_mean, scale=1.0) for z_mean in est.Z_mu]

        obs_ll = []
        rep_ll = []

        # Monte-Carlo estimation of expected log probability
        pbar = tqdm(total=n_eval * x_valid.shape[0], disable=silent)
        for j in range(n_eval):
            w_sample = w_rv.rvs()

            obs_ll_samples = []
            rep_ll_samples = []
            for i in range(x_valid.shape[0]):
                z_sample = np.atleast_1d(z_rv[i].rvs())

                holdout_mean = w_sample.dot(z_sample) * self.holdout_mask_[i]

                x_rv = stats.norm(loc=holdout_mean, scale=est.sigma_x)
                # mean over features
                obs_ll_samples.append(np.mean(x_rv.logpdf(x_valid[i]), axis=0))
                rep_ll_samples.append(np.mean(x_rv.logpdf(holdout_gen[:, i, :]), axis=1))
                pbar.update()

            obs_ll.append(np.stack(obs_ll_samples))
            rep_ll.append(np.stack(rep_ll_samples))

        obs_ll = np.stack(obs_ll)
        rep_ll = np.stack(rep_ll)

        # mean over evaluations
        obs_ll_per_zi = np.mean(obs_ll, axis=0)
        rep_ll_per_zi = np.mean(rep_ll, axis=0)

        # average over posterior draws
        pvals = np.mean(rep_ll_per_zi < obs_ll_per_zi[:, np.newaxis], axis=1)
        # average over subjects
        overall_pval = np.mean(pvals[self.holdout_subjects_])

        return overall_pval


class BaseLinearModel(Model, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        stan_class: type,
        scale_alpha: float,
        scale_beta: float,
        standardize: bool,
        posterior_samples: int,
        n_chains: int,
        random_state: RandomStateType,
        n_jobs: int,
    ) -> None:
        super().__init__(
            posterior_samples=posterior_samples,
            stan_class=stan_class,
            random_state=random_state,
        )
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.standardize = standardize
        self.n_chains = n_chains
        self.n_jobs = n_jobs

    def _get_data(self, X, y, X_valid, y_valid):
        """Return data for StanModel.sampling"""
        if self.standardize:
            self._x_scaler = StandardScaler()
            Xt = self._x_scaler.fit_transform(X)
        else:
            Xt = check_array(X, copy=True)

        if X_valid is None:
            N_tilde = 0
            Xt_valid = np.empty((0, X.shape[1]), dtype=X.dtype)
            y_valid = np.empty(0, dtype=y.dtype)
            do_simulate = 0
        else:
            if self.standardize:
                Xt_valid = self._x_scaler.transform(X_valid)
            else:
                Xt_valid = X_valid.copy()
            N_tilde = Xt_valid.shape[0]
            do_simulate = 1

        stan_data = {
            'N': Xt.shape[0],
            'K': Xt.shape[1],
            'X': Xt,
            'y': y,
            'scale_alpha': self.scale_alpha,
            'scale_beta': self.scale_beta,
            'N_tilde': N_tilde,
            'X_tilde': Xt_valid,
            'y_tilde': y_valid,
            'do_simulate': do_simulate,
        }
        return stan_data

    def _fit(self, stan_data):
        seed = pystan.misc._check_seed(self.random_state)
        fit = self._stan_class(stan_data, seed)

        pars = fit._get_param_names()

        stan_args = {
            "chains": self.n_chains,
            "iter": 2 * self.posterior_samples,
            "warmup": self.posterior_samples,
            "thin": 1,
            "init": "random",
            "seed": seed,
            "method": "sampling",
            "algorithm": "NUTS",
            "sample_file": None,
            "diagnostic_file": None,
            "control": None,
        }
        warmup = stan_args["warmup"]
        thin = stan_args["thin"]
        chains = stan_args["chains"]
        iterations = stan_args["iter"]

        chain_args = pystan.misc._config_argss(**stan_args)

        module = importlib.import_module(self._stan_class.__module__)
        ret_and_samples = Parallel(n_jobs=self.n_jobs, max_nbytes=1e9)(
            delayed(module._call_sampler)(stan_data, args, pars) for args in chain_args
        )

        # ret_and_samples = fit._call_sampler(stan_args, pars_oi=pars)
        samples = [smpl for _, smpl in ret_and_samples]

        # number of samples saved after thinning
        warmup2 = 1 + (warmup - 1) // thin
        n_kept = 1 + (iterations - warmup - 1) // thin
        n_save = n_kept + warmup2

        random_state = np.random.RandomState(seed)
        perm_lst = [random_state.permutation(int(n_kept)) for _ in range(chains)]
        fnames_oi = fit._get_param_fnames_oi()
        n_flatnames = len(fnames_oi)
        fit.sim = {
            'samples': samples,
            # rstan has this; name clashes with 'chains' in samples[0]['chains']
            'chains': len(samples),
            'iter': iterations,
            'warmup': warmup,
            'thin': thin,
            'n_save': [n_save] * chains,
            'warmup2': [warmup2] * chains,
            'permutation': perm_lst,
            'pars_oi': fit._get_param_names_oi(),
            'dims_oi': fit._get_param_dims_oi(),
            'fnames_oi': fnames_oi,
            'n_flatnames': n_flatnames,
        }
        fit.model_name = self.__class__.__name__
        fit.model_pars = pars
        fit.par_dims = fit._get_param_dims()
        fit.mode = 0

        samples = fit.extract(pars=["alpha", "beta"])

        alpha = samples["alpha"][np.newaxis]
        beta = samples["beta"].T
        if self.standardize:
            # rescale coefficients
            beta = beta / self._x_scaler.scale_[:, np.newaxis]
            alpha = alpha - np.dot(self._x_scaler.mean_[np.newaxis, :], beta)

        self._results = {"alpha": alpha, "beta": beta}
        return self

    def get_mean(self, name: str) -> np.ndarray:
        return self.get_samples(name).mean(axis=1)

    def get_samples(self, name: str) -> np.ndarray:
        return self._results[name]

    @property
    def coef_(self):
        return self.get_mean("beta")

    @property
    def intercept_(self):
        return self.get_mean("alpha")

    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            X_valid: Optional[np.ndarray] = None,
            y_valid: Optional[np.ndarray] = None) -> object:
        """Fit model"""
        stan_data = self._get_data(X, y, X_valid, y_valid)
        return self._fit(stan_data)

    def summary(self, names=None):
        coefs = self.get_posterior_samples(names)
        df = pd.DataFrame({
            "mean": coefs.mean(axis=1),
            "sd": coefs.std(ddof=1, axis=1),
        })
        return df

    def get_posterior_samples(self, names=None) -> pd.DataFrame:
        alpha = self.get_samples("alpha")
        beta = self.get_samples("beta")

        if names is None:
            names = [f"beta.{i}" for i in range(beta.shape[0])]
        else:
            names = list(names)
            assert len(names) == beta.shape[0], "{} != {}".format(len(names), beta.shape[0])

        index = ["Intercept"] + names
        samples = pd.DataFrame(np.row_stack([alpha, beta]), index=index)
        return samples


class LogReg(BaseLinearModel):
    """Logistic Regression"""

    def __init__(
        self,
        scale_alpha: float = 5.0,
        scale_beta: float = 5.0,
        standardize: bool = True,
        posterior_samples: int = 1000,
        n_chains: int = 4,
        random_state: RandomStateType = None,
        n_jobs: int = 1,
    ) -> None:
        from ._logreg import StanFit4Model

        super().__init__(
            stan_class=StanFit4Model,
            scale_alpha=scale_alpha,
            scale_beta=scale_beta,
            standardize=standardize,
            posterior_samples=posterior_samples,
            n_chains=n_chains,
            random_state=random_state,
            n_jobs=n_jobs,
        )


InferredBPMFParameters = namedtuple(
    "InferredBPMFParameters",
    ("U", "V", "mu_u", "mu_v", "cov_u", "cov_v"),
)


class BPMF(Model):
    """Bayesian probabilistic matrix factorization"""

    def __init__(
        self,
        latent_dim: int,
        standardize: bool = True,
        posterior_samples: int = 1000,
        random_state: RandomStateType = None,
    ) -> None:
        from ._bpmf_reparameterized import StanFit4Model

        super().__init__(
            posterior_samples=posterior_samples,
            stan_class=StanFit4Model,
            random_state=random_state)
        self.latent_dim = latent_dim
        self.standardize = standardize
        self.beta_0 = 2
        self.rating_std = 0.5

    def _get_data(self, X, y):
        """Return data for StanModel.sampling"""
        if self.standardize:
            Xt = StandardScaler().fit_transform(X)
        else:
            Xt = X.copy()

        self.holdout_mask_, self.holdout_subjects_ = make_holdout_mask(
            n_samples=Xt.shape[0],
            n_features=Xt.shape[1],
            random_state=self.random_state)
        self.n_samples_, self.n_features_ = Xt.shape
        self.X_valid_ = Xt * self.holdout_mask_

        rows, cols = (self.holdout_mask_ == 0).nonzero()

        stan_data = {
            "n_users": self.n_samples_,
            "n_items": self.n_features_,
            "n_obs": rows.shape[0],
            "the_rank": self.latent_dim,
            "obs_users": rows + 1,  # indices start at 1
            "obs_items": cols + 1,
            "obs_ratings": Xt[(rows, cols)],
            "rating_std": self.rating_std,
            "mu_0": np.zeros(self.latent_dim),
            "beta_0": self.beta_0,  # observation noise precision
            "nu_0": self.latent_dim,  # degrees of freedom
            # "w_0": np.eye(self.latent_dim),  # scale matrix, PSD
        }
        return stan_data

    def replicated_data(self) -> np.ndarray:
        """Retrieve data generated from the predictive distribution

        Returns
        -------
        X_rep : np.ndarray, shape = (n_draws, n_samples, n_features)
        """
        U_rep = self.get_samples("U")
        # make first dimension the number of posterior samples
        U_rep = np.stack(
            [U_rep[..., i].reshape((self.n_samples_, self.latent_dim), order="F")
             for i in range(U_rep.shape[-1])]
        )

        V_rep = self.get_samples("V")
        V_rep = np.stack(
            [V_rep[..., i].reshape((self.n_features_, self.latent_dim), order="F")
             for i in range(V_rep.shape[-1])]
        )

        # X_rep = np.matmul(U_rep, np.transpose(V_rep, [0, 2, 1]))
        X_rep = np.einsum('aij,akj->aik', U_rep, V_rep)
        # look only at the heldout entries
        holdout_gen = np.multiply(X_rep, self.holdout_mask_[np.newaxis])
        return holdout_gen

    def _get_cholesky_cov_samples(self, c_u, z_u):
        c_u = self.get_samples("c_u")
        z_u = self.get_samples("z_u")
        latent_dim = self.latent_dim
        A_u_inv = np.empty((c_u.shape[-1], latent_dim, latent_dim))
        tril_idx = np.tril_indices(latent_dim, -1)
        diag_idx = np.diag_indices(latent_dim)
        A_u = np.zeros((latent_dim, latent_dim))
        for i in range(c_u.shape[-1]):
            A_u[tril_idx] = z_u[:, i]
            A_u[diag_idx] = np.sqrt(c_u[:, i])
            A_u_inv[i, ...] = np.linalg.inv(A_u)

        return A_u_inv

    def _get_cholesky_cov_mean(self, c_u, z_u):
        latent_dim = self.latent_dim
        A_u = np.zeros((latent_dim, latent_dim))
        tril_idx = np.tril_indices(latent_dim, -1)
        diag_idx = np.diag_indices(latent_dim)
        A_u[tril_idx] = z_u
        A_u[diag_idx] = np.sqrt(c_u)
        A_u_inv = np.linalg.inv(A_u)

        return A_u_inv

    def _get_mu_mean(self, cov_mean, alpha):
        # mu_0 + A_u_inv * alpha_u;
        return np.dot(cov_mean, alpha)

    def posterior_mean_estimates(self) -> InferredBPMFParameters:
        """Retrieve posterior mean estimates."""
        latent_dim = self.latent_dim
        u_mean = self.get_mean("U").reshape((-1, latent_dim), order="F")
        v_mean = self.get_mean("V").reshape((-1, latent_dim), order="F")
        cov_u_mean = self._get_cholesky_cov_mean(self.get_mean("c_u"), self.get_mean("z_u"))
        cov_v_mean = self._get_cholesky_cov_mean(self.get_mean("c_v"), self.get_mean("z_v"))
        mu_u_mean = self._get_mu_mean(cov_u_mean, self.get_mean("alpha_u"))
        mu_v_mean = self._get_mu_mean(cov_v_mean, self.get_mean("alpha_v"))

        return InferredBPMFParameters(
                u_mean,
                v_mean,
                mu_u_mean,
                mu_v_mean,
                cov_u_mean,
                cov_v_mean)

    def check_model(self, n_eval: int = 100, silent: bool = False) -> float:
        """Posterior predictive check aka Bayesian p-value"""
        x_valid = self.X_valid_

        holdout_gen = self.replicated_data()
        est = self.posterior_mean_estimates()

        u_rv = stats.multivariate_normal(mean=est.mu_u, cov=est.cov_u / self.beta_0)
        v_rv = stats.multivariate_normal(mean=est.mu_v, cov=est.cov_v / self.beta_0)

        obs_ll = []
        rep_ll = []

        # Monte-Carlo estimation of expected log probability
        pbar = tqdm(total=n_eval, disable=silent)
        for j in range(n_eval):
            V = v_rv.rvs(size=self.n_features_)
            if V.ndim == 1:
                V = V[:, np.newaxis]
            U = u_rv.rvs(size=self.n_samples_)
            if U.ndim == 1:
                U = U[:, np.newaxis]
            holdout_mean = np.dot(U, V.T) * self.holdout_mask_

            x_rv = stats.norm(loc=holdout_mean, scale=self.rating_std)

            # mean over features
            obs_ll.append(np.mean(x_rv.logpdf(x_valid), axis=1))
            rep_ll.append(np.mean(x_rv.logpdf(holdout_gen), axis=2).T)
            pbar.update()

        obs_ll = np.stack(obs_ll)
        rep_ll = np.stack(rep_ll)

        # mean over evaluations
        obs_ll_per_zi = np.mean(obs_ll, axis=0)
        rep_ll_per_zi = np.mean(rep_ll, axis=0)

        # average over posterior draws
        pvals = np.mean(rep_ll_per_zi < obs_ll_per_zi[:, np.newaxis], axis=1)
        # average over subjects
        overall_pval = np.mean(pvals[self.holdout_subjects_])

        return overall_pval

@dataclass
class InferredPPCAParameters:
    W_mu: np.ndarray
    Z_mu: np.ndarray
    sigma_x: float
    alpha_ov: np.ndarray
    sigma_alpha: float


class PPCAExtended(PPCA):
    def __init__(self,
                 known_causes: Sequence[str],
                 latent_dim: int,
                 sigma_w: float = 2.0,
                 posterior_samples: int = 1000,
                 random_state: RandomStateType = None) -> None:
        super().__init__(
            latent_dim=latent_dim,
            sigma_w=sigma_w,
            standardize=False,
            posterior_samples=posterior_samples,
            random_state=random_state)
        from ._ppca4_adni import StanFit4Model

        self._stan_class = StanFit4Model
        self.known_causes = known_causes

    def _get_data(self, X, y):
        """Return data for StanModel.sampling"""
        others = check_array(X.loc[:, self.known_causes], copy=True)
        Xt = check_array(X.drop(self.known_causes, axis=1), copy=True)

        self.holdout_mask_, self.holdout_subjects_ = make_holdout_mask(
            n_samples=Xt.shape[0],
            n_features=Xt.shape[1],
            random_state=self.random_state)
        self.n_samples_, self.n_features_ = Xt.shape
        self.X_valid_ = Xt * self.holdout_mask_
        self.X_others_ = others

        stan_data = {
            'N': Xt.shape[0],
            'D': Xt.shape[1],
            'DZ': self.latent_dim,
            'P': others.shape[1],
            'X': Xt,
            'other_vars': others,
            'sigma_w': self.sigma_w,
            'holdout': self.holdout_mask_,
        }
        return stan_data

    def posterior_mean_estimates(self) -> InferredPPCAParameters:
        """Retrieve posterior mean estimates."""
        latent_dim = self.latent_dim
        w_mean = self.get_mean("W").reshape((-1, latent_dim), order="F")
        z_mean = self.get_mean("Z").reshape((latent_dim, -1), order="F").T
        sigma_x_mean = self.get_mean("sigma_x", False)
        alpha_ov_mean = self.get_mean("alpha_ov").reshape((-1, len(self.known_causes)), order="F")
        sigma_alpha_mean = self.get_mean("sigma_alpha", False)

        return InferredPPCAParameters(
            w_mean, z_mean, sigma_x_mean, alpha_ov_mean, sigma_alpha_mean
        )

    def check_model(self, n_eval: int = 100, silent: bool = False) -> float:
        """Posterior predictive check aka Bayesian p-value"""
        x_valid = self.X_valid_

        holdout_gen = self.replicated_data()
        est = self.posterior_mean_estimates()

        w_rv = stats.norm(loc=est.W_mu, scale=self.sigma_w)
        z_rv = [stats.norm(loc=z_mean, scale=1.0) for z_mean in est.Z_mu]
        alpha_rv = stats.norm(loc=est.alpha_ov, scale=est.sigma_alpha)  # shape = [n_features, n_known_causes]

        obs_ll = []
        rep_ll = []

        # Monte-Carlo estimation of expected log probability
        pbar = tqdm(total=n_eval * x_valid.shape[0], disable=silent)
        for j in range(n_eval):
            w_sample = w_rv.rvs()
            alpha_sample = alpha_rv.rvs()

            obs_ll_samples = []
            rep_ll_samples = []
            for i in range(x_valid.shape[0]):
                z_sample = np.atleast_1d(z_rv[i].rvs())

                holdout_mean = (w_sample.dot(z_sample) + alpha_sample.dot(self.X_others_[i])) * self.holdout_mask_[i]

                x_rv = stats.norm(loc=holdout_mean, scale=est.sigma_x)
                # mean over features
                obs_ll_samples.append(np.mean(x_rv.logpdf(x_valid[i]), axis=0))
                rep_ll_samples.append(np.mean(x_rv.logpdf(holdout_gen[:, i, :]), axis=1))
                pbar.update()

            obs_ll.append(np.stack(obs_ll_samples))
            rep_ll.append(np.stack(rep_ll_samples))

        obs_ll = np.stack(obs_ll)
        rep_ll = np.stack(rep_ll)

        # mean over evaluations
        obs_ll_per_zi = np.mean(obs_ll, axis=0)
        rep_ll_per_zi = np.mean(rep_ll, axis=0)

        # average over posterior draws
        pvals = np.mean(rep_ll_per_zi < obs_ll_per_zi[:, np.newaxis], axis=1)
        # average over subjects
        overall_pval = np.mean(pvals[self.holdout_subjects_])

        return overall_pval

    def mean_reconstruction(self, z: np.ndarray) -> np.ndarray:
        assert z.shape == (self.n_samples_, self.latent_dim)

        w = self.get_mean("W").reshape((-1, self.latent_dim), order="F")
        alpha = self.get_mean("alpha_ov").reshape((-1, len(self.known_causes)), order="F")

        recon = z.dot(w.T) + self.X_others_.dot(alpha.T)
        return recon


@dataclass
class InferredBPMFParameters:
    U: np.ndarray
    V: np.ndarray
    mu_u: np.ndarray
    mu_v: np.ndarray
    cov_u: np.ndarray
    cov_v: np.ndarray
    sigma_alpha: float
    alpha_ov: np.ndarray


class BPMFExtended(BPMF):
    def __init__(self,
                 known_causes: Sequence[str],
                 latent_dim: int,
                 posterior_samples: int = 1000,
                 random_state=None) -> None:
        super().__init__(
            latent_dim=latent_dim,
            standardize=False,
            posterior_samples=posterior_samples,
            random_state=random_state)
        from ._bpmf_reparameterized_adni import StanFit4Model

        self._stan_class = StanFit4Model
        self.known_causes = known_causes

    def _get_data(self, X, y):
        """Return data for StanModel.sampling"""
        others = check_array(X.loc[:, self.known_causes], copy=True)
        Xt = check_array(X.drop(self.known_causes, axis=1), copy=True)

        self.holdout_mask_, self.holdout_subjects_ = make_holdout_mask(
            n_samples=Xt.shape[0],
            n_features=Xt.shape[1],
            random_state=self.random_state)
        self.n_samples_, self.n_features_ = Xt.shape
        self.X_valid_ = Xt * self.holdout_mask_
        self.X_others_ = others

        rows, cols = (self.holdout_mask_ == 0).nonzero()

        stan_data = {
            "n_users": self.n_samples_,
            "n_items": self.n_features_,
            "n_obs": rows.shape[0],
            "P": others.shape[1],
            "the_rank": self.latent_dim,
            "obs_users": rows + 1,  # indices start at 1
            "obs_items": cols + 1,
            "obs_ratings": Xt[(rows, cols)],
            "other_vars": others,
            "rating_std": self.rating_std,
            "mu_0": np.zeros(self.latent_dim),
            "beta_0": self.beta_0,  # observation noise precision
            "nu_0": self.latent_dim,  # degrees of freedom
            # "w_0": np.eye(self.latent_dim),  # scale matrix, PSD
        }
        return stan_data

    def _reconstruction_mean(self) -> np.ndarray:
        # Z.T @ W.T
        mu = super()._reconstruction_mean()

        alpha = self.get_samples("alpha_ov")
        # make first dimension the number of posterior samples
        alpha = np.stack(
            [alpha[:, i].reshape((self.n_features_, len(self.known_causes)), order="F")
            for i in range(alpha.shape[1])]
        )

        # X @ alpha
        mu += np.einsum("ik,ajk->aij", self.X_others_, alpha)

        return mu

    def posterior_mean_estimates(self) -> InferredBPMFParameters:
        """Retrieve posterior mean estimates."""
        latent_dim = self.latent_dim
        u_mean = self.get_mean("U").reshape((-1, latent_dim), order="F")
        v_mean = self.get_mean("V").reshape((-1, latent_dim), order="F")
        cov_u_mean = self._get_cholesky_cov_mean(self.get_mean("c_u"), self.get_mean("z_u"))
        cov_v_mean = self._get_cholesky_cov_mean(self.get_mean("c_v"), self.get_mean("z_v"))
        mu_u_mean = self._get_mu_mean(cov_u_mean, self.get_mean("alpha_u"))
        mu_v_mean = self._get_mu_mean(cov_v_mean, self.get_mean("alpha_v"))
        sigma_alpha = self.get_mean("sigma_alpha", False)
        alpha_ov = self.get_mean("alpha_ov").reshape(-1, len(self.known_causes), order="F")

        return InferredBPMFParameters(
                u_mean,
                v_mean,
                mu_u_mean,
                mu_v_mean,
                cov_u_mean,
                cov_v_mean,
                sigma_alpha,
                alpha_ov,
        )

    def replicated_data(self) -> np.ndarray:
        """Retrieve data generated from the predictive distribution

        Returns
        -------
        X_rep : np.ndarray, shape = (n_draws, n_samples, n_features)
        """
        X_rep = self._reconstruction_samples()
        # look only at the heldout entries
        holdout_gen = np.multiply(X_rep, self.holdout_mask_[np.newaxis])
        return holdout_gen

    def mean_reconstruction(self, U: np.ndarray) -> np.ndarray:
        assert U.shape == (self.n_samples_, self.latent_dim)

        V = self.get_mean("V").reshape((-1, self.latent_dim), order="F")
        alpha = self.get_mean("alpha_ov").reshape(-1, len(self.known_causes), order="F")

        recon = np.dot(U, V.T) + self.X_others_.dot(alpha.T)
        return recon

    def _reconstruction_samples(self) -> np.ndarray:
        U_rep = self.get_samples("U")
        # make first dimension the number of posterior samples
        U_rep = np.stack(
            [U_rep[:, i].reshape((self.n_samples_, self.latent_dim), order="F")
             for i in range(U_rep.shape[1])]
        )

        V_rep = self.get_samples("V")
        V_rep = np.stack(
            [V_rep[:, i].reshape((self.n_features_, self.latent_dim), order="F")
             for i in range(V_rep.shape[1])]
        )

        alpha_rep = self.get_samples("alpha_ov")
        alpha_rep = np.stack([
            alpha_rep[:, i].reshape(-1, len(self.known_causes), order="F")
            for i in range(alpha_rep.shape[1])
        ])
        # np.matmul(self.X_others_, np.swapaxes(alpha_rep, 2, 1))
        lin_pred = np.einsum('ij,akj->aik', self.X_others_, alpha_rep)

        # X_rep = np.matmul(U_rep, np.transpose(V_rep, [0, 2, 1]))
        X_rep = lin_pred + np.einsum('aij,akj->aik', U_rep, V_rep)
        return stats.norm.rvs(loc=X_rep, scale=self.rating_std, random_state=self.random_state)

    def check_model(self, n_eval: int = 100, silent: bool = False) -> float:
        """Posterior predictive check aka Bayesian p-value"""
        x_valid = self.X_valid_

        holdout_gen = self.replicated_data()
        est = self.posterior_mean_estimates()

        u_rv = stats.multivariate_normal(mean=est.mu_u, cov=est.cov_u / self.beta_0)
        v_rv = stats.multivariate_normal(mean=est.mu_v, cov=est.cov_v / self.beta_0)
        alpha_rv = stats.norm(loc=est.alpha_ov, scale=est.sigma_alpha)  # shape = [n_features, n_known_causes]

        obs_ll = []
        rep_ll = []

        # Monte-Carlo estimation of expected log probability
        pbar = tqdm(total=n_eval, disable=silent)
        for j in range(n_eval):
            V = v_rv.rvs(size=self.n_features_)
            if V.ndim == 1:
                V = V[:, np.newaxis]
            U = u_rv.rvs(size=self.n_samples_)
            if U.ndim == 1:
                U = U[:, np.newaxis]

            alpha_sample = alpha_rv.rvs()
            holdout_mean = (np.dot(U, V.T) + np.dot(self.X_others_, alpha_sample.T)) * self.holdout_mask_

            x_rv = stats.norm(loc=holdout_mean, scale=self.rating_std)

            # mean over features
            obs_ll.append(np.mean(x_rv.logpdf(x_valid), axis=1))
            rep_ll.append(np.mean(x_rv.logpdf(holdout_gen), axis=2).T)
            pbar.update()

        obs_ll = np.stack(obs_ll)
        rep_ll = np.stack(rep_ll)

        # mean over evaluations
        obs_ll_per_zi = np.mean(obs_ll, axis=0)
        rep_ll_per_zi = np.mean(rep_ll, axis=0)

        # average over posterior draws
        pvals = np.mean(rep_ll_per_zi < obs_ll_per_zi[:, np.newaxis], axis=1)
        # average over subjects
        overall_pval = np.mean(pvals[self.holdout_subjects_])

        return overall_pval
