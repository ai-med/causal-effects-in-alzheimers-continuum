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
// Binomial Linear Model with logit link function
data {
  // number of observations
  int<lower=1> N;
  // response
  int<lower=0,upper=1> y[N];
  // number of columns in the design matrix X
  int<lower=1> K;
  // design matrix X
  // should not include an intercept
  matrix [N, K] X;
  // priors on alpha
  real<lower=0> scale_alpha;
  real<lower=0> scale_beta;
  // keep responses
  int<lower=0> N_tilde;
  int<lower=0,upper=1> y_tilde[N_tilde];
  matrix[N_tilde, K] X_tilde;
  int<lower=0,upper=1> do_simulate;
}
parameters {
  // regression coefficient vector
  real alpha;
  vector[K] beta;
}
transformed parameters {
  vector[N] mu;
  vector[N_tilde] mu_tilde;

  mu = alpha + X * beta;
  if (do_simulate) {
    mu_tilde = alpha + X_tilde * beta;
  }
}
model {
  // priors
  alpha ~ normal(0., scale_alpha);
  beta ~ normal(0., scale_beta);
  // likelihood
  y ~ bernoulli_logit(mu);
}
generated quantities {
  if (do_simulate) {
    // simulate data from the posterior
    int y_rep[N_tilde];
    // log-likelihood posterior
    vector[N_tilde] log_lik;
    for (i in 1:num_elements(y_rep)) {
      y_rep[i] = bernoulli_logit_rng(mu_tilde[i]);
      log_lik[i] = bernoulli_logit_lpmf(y_tilde[i] | mu_tilde[i]);
    }
  }
}
