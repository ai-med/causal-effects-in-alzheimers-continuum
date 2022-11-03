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
// Adapted from https://github.com/autonlab/active-matrix-factorization/blob/master/stan-bpmf/bpmf_w0identity.stan
data {
  int<lower=1> n_users;
  int<lower=1> n_items;
  int<lower=1> P;  // number of other variables
  vector[P] other_vars[n_users]; // age, age^2, edu, sex, ptau

  int<lower=1,upper=min(n_users,n_items)> the_rank;

  // observed data
  int<lower=1,upper=n_users*n_items> n_obs;
  int<lower=1,upper=n_users> obs_users[n_obs];
  int<lower=1,upper=n_items> obs_items[n_obs];
  real obs_ratings[n_obs];

  // fixed hyperparameters
  real<lower=0> rating_std; // observation noise std deviation, usually 1/2

  vector[the_rank] mu_0; // mean for feature means, usually zero

  // feature mean covariances are beta_0 * inv wishart(nu_0, w_0)
  real<lower=0> beta_0; // usually 2
  int<lower=the_rank> nu_0; // deg of freedom, usually == rank
  // cov_matrix[the_rank] w_0; // scale matrix, usually identity
}

transformed data {
  matrix[the_rank, the_rank] eye;
  vector<lower=0>[the_rank] nu_0_minus_i;

  for (i in 1:the_rank) {
    for (j in 1:the_rank) {
      eye[i, j] = 0.0;
    }
    nu_0_minus_i[i] = nu_0 - i + 1;
    eye[i, i] = 1.0;
  }
}

parameters {
  // regression weights
  real<lower=0> sigma_alpha;
  vector[P] alpha_ov[n_items];

  // latent factors
  vector[the_rank] U[n_users];
  vector[the_rank] V[n_items];

  // means and covs on latent factors
  vector[the_rank] alpha_u;
  vector<lower=0>[the_rank] c_u;
  vector[max((the_rank * (the_rank - 1)) / 2, 1)] z_u;

  vector[the_rank] alpha_v;
  vector<lower=0>[the_rank] c_v;
  vector[max((the_rank * (the_rank - 1)) / 2, 1)] z_v;
}

transformed parameters {
  vector[n_obs] obs_means;
  for (n in 1:n_obs) {
    obs_means[n] = dot_product(U[obs_users[n]], V[obs_items[n]]) + dot_product(alpha_ov[obs_items[n]], other_vars[obs_users[n]]);
  }
}

model {
  // from https://mc-stan.org/docs/2_25/stan-users-guide/reparameterization-section.html#multivariate-reparameterizations
  // W^-1 ~ inv_wishart(nu_0, w_0);
  // y ~ multi_normal(mu, W^-1);
  // Reparameterization:
  // W^-1 = L^T^-1 * A^T^-1 * A^-1 * L^-1;
  // y ~ multi_normal_cholesky(mu, A^-1 * L^-1);
  // If w_0 = Identity; ==> L = Identity;
  // W^-1 = A^T^-1 * A^-1;
  // y ~ multi_normal_cholesky(mu, A^-1);

  // means on latent factors
  vector[the_rank] mu_u;
  vector[the_rank] mu_v;
  // covs on latent factors
  matrix[the_rank, the_rank] A_u;  // lower triangular
  matrix[the_rank, the_rank] A_u_inv;
  matrix[the_rank, the_rank] A_v;  // lower triangular
  matrix[the_rank, the_rank] A_v_inv;
  int count;

  // regression coefficients
  sigma_alpha ~ cauchy(0, 2.5);

  // hyperpriors on latent factor hyperparams
  count = 1;
  for (j in 1:(the_rank-1)) {
    for (i in (j+1):the_rank) {
      A_u[i, j] = z_u[count];
      A_v[i, j] = z_v[count];
      count += 1;
    }
    for (i in 1:(j - 1)) {
      A_u[i, j] = 0.0;
      A_v[i, j] = 0.0;
    }
    A_u[j, j] = sqrt(c_u[j]);
    A_v[j, j] = sqrt(c_v[j]);
  }
  for (i in 1:(the_rank-1)) {
    A_u[i, the_rank] = 0;
    A_v[i, the_rank] = 0;
  }
  A_u[the_rank, the_rank] = sqrt(c_u[the_rank]);
  A_v[the_rank, the_rank] = sqrt(c_v[the_rank]);

  A_u_inv = mdivide_left_tri_low(A_u, eye);
  A_v_inv = mdivide_left_tri_low(A_v, eye);

  c_u ~ chi_square(nu_0_minus_i);
  c_v ~ chi_square(nu_0_minus_i);
  z_u ~ std_normal();
  z_v ~ std_normal();

  // Implies:
  //   cov_u ~ inv_wishart(nu_0, I);
  //   mu_u ~ multi_normal(mu_0, cov_u / beta_0);
  alpha_u ~ normal(0, 1.0 / beta_0);
  alpha_v ~ normal(0, 1.0 / beta_0);
  mu_u = mu_0 + A_u_inv * alpha_u;
  mu_v = mu_0 + A_v_inv * alpha_v;

  // prior on latent factors
  // Implies:
  //   U[i] ~ multi_normal(mu_u, cov_u);
  for (i in 1:n_users)
    U[i] ~ multi_normal_cholesky(mu_u, A_u_inv);
  for (j in 1:n_items) {
    V[j] ~ multi_normal_cholesky(mu_v, A_v_inv);
    // regression coefficients
    alpha_ov[j] ~ normal(0, sigma_alpha);
  }

  // observed data likelihood
  obs_ratings ~ normal(obs_means, rating_std);
}

/*
generated quantities {
  real training_rmse;
  training_rmse = 0;
  for (i in 1:n_obs) {
    training_rmse = training_rmse
      + square(predictions[obs_users[i], obs_items[i]] - obs_ratings[i]);
  }
  training_rmse = sqrt(training_rmse);
}
*/
