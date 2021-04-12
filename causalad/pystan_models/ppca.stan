// This file is part of Estimation of Causal Effects in the Alzheimer's Continuum (Causal-AD).
//
// Causal-AD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Causal-AD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Causal-AD. If not, see <https://www.gnu.org/licenses/>.
// Adapted from https://github.com/RSNirwan/HouseholderBPCA/blob/master/py_stan_code/ppca.stan
data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> DZ;
  int<lower=1> P;  // number of other variables
  vector[D] X[N];
  vector[P] other_vars[N]; // age, age^2, edu, sex, ptau
  real<lower=0> sigma_w;
  vector[D] holdout[N];
}
transformed data{
  vector[D] X_sel[N];
  vector[D] mu;
  matrix[D, D] Sigma;
  vector[DZ] mu_Z;
  matrix[DZ, DZ] Sigma_Z;

  mu = rep_vector(0.0, D);
  for (i in 1:D) {
    for (j in 1:D) {
      Sigma[i, j] = 0;
    }
    Sigma[i, i] = 1;
  }

  mu_Z = rep_vector(0.0, DZ);
  for (i in 1:DZ) {
    for (j in 1:DZ) {
      Sigma_Z[i, j] = 0;
    }
    Sigma_Z[i, i] = 1;
  }

  for (i in 1:N) {
    X_sel[i] = X[i] .* (1.0 - holdout[i]);
  }
}
parameters {
  matrix[D, DZ] W;
  real<lower=0> sigma_x;
  real<lower=0> sigma_alpha;
  matrix[D, P] alpha_ov;
}
transformed parameters {
  cholesky_factor_cov[D] L;
  vector[DZ] Z_mu[N];
  matrix[DZ, DZ] Z_cov;

  {
    //matrix[D, D] K = W*W'; // tcrossprod(matrix x)
    matrix[D, D] K = tcrossprod(W);
    for (d in 1:D)
      K[d,d] += square(sigma_x) + 1e-14;
    L = cholesky_decompose(K);
  }
  // Compute mean and covariance of posterior distribution p(Z|X) ~ Normal(Z_mu, Z_cov)
  {
    matrix[DZ, DZ] M;
    matrix[DZ, DZ] M_inv;
    matrix[DZ, D] Z_mu_l;

    M = crossprod(W);
    for (d in 1:DZ)
      M[d,d] += square(sigma_x) + 1e-14;
    M_inv = inverse(M);
    Z_mu_l = M_inv * W';
    for (i in 1:N)
      Z_mu[i] = Z_mu_l * X_sel[i];
    Z_cov = M / square(sigma_x);
  }
}
model {
  //mu ~ normal(0, 10);
  //sigma_x ~ normal(0, 1);
  sigma_x ~ cauchy(0, 2.5);
  to_vector(W) ~ normal(0, sigma_w);
  sigma_alpha ~ cauchy(0, 2.5);
  to_vector(alpha_ov) ~ normal(0, sigma_alpha);

  for (i in 1:N) {
    X_sel[i] ~ multi_normal_cholesky(mu + (alpha_ov * other_vars[i]), L);
  }
}
generated quantities {
  vector[D] X_rep[N];
  matrix[D, DZ] Ws;
  matrix[D, P] alpha_s;

  // draw principal axes
  for (j in 1:D) {
    for (k in 1:DZ) {
      Ws[j,k] = normal_rng(0, sigma_w);
    }
    for (k in 1:P) {
      alpha_s[j,k] = normal_rng(0, sigma_alpha);
    }
  }

  for (i in 1:N) {
    vector[DZ] Zs;

    // draw latent vector
    Zs = to_vector(multi_normal_rng(mu_Z, Sigma_Z));
    // produce replicated data
    X_rep[i] = multi_normal_rng((Ws * Zs + alpha_s * other_vars[i]) .* (1.0 - holdout[i]), sigma_w * Sigma);
  }
}
