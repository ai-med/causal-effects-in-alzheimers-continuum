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
  for (i in 1:N) {
    X_sel[i] = X[i] .* (1.0 - holdout[i]);
  }
}
parameters {
  matrix[D, DZ] W;
  matrix[DZ, N] Z;
  real<lower=0> sigma_x;
  real<lower=0> sigma_alpha;
  matrix[D, P] alpha_ov;
}
model {
  sigma_x ~ cauchy(0, 2.5);
  to_vector(W) ~ normal(0, sigma_w);
  to_vector(Z) ~ std_normal();
  sigma_alpha ~ cauchy(0, 2.5);
  to_vector(alpha_ov) ~ normal(0, sigma_alpha);

  // likelihood
  for (i in 1:N) {
    X_sel[i] ~ normal((W * col(Z, i) + alpha_ov * other_vars[i]) .* (1.0 - holdout[i]), sigma_x);
  }
}
