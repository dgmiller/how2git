// HBMNL for discrete choice experiments
data {
  int<lower=2> A; // number of alternatives (choices) per question
  int<lower=1> L; // number of feature variables
  int<lower=1> R; // number of respondents
  int<lower=1> T; // number of questions (unique inquiries)
  int<lower=1> C; // number of respondent covariates (demographics, etc)

  matrix[A, L] X[R, T]; // matrix of attributes for each obs
  int<lower=1, upper=A> Y[R, T]; // observed responses
  matrix[R, C] Z; // vector of covariates for each respondent

  real mu_mean;
  real alpha_mean;
  real<lower=0> mu_scale;
  real<lower=0> alpha_scale;
  real<lower=0> lkj_param;
}

parameters {
  matrix[L, R] alpha; // prior on variance of utilities B
  vector<lower=0, upper=pi()/2>[L] tau_unif;
  matrix[C, L] mu; // prior on mean of utilities B
  cholesky_factor_corr[L] L_Omega;

}

transformed parameters {
  matrix[R, L] B; // matrix of beta coefficients
  vector<lower=0>[L] tau; // prior scale
  for (l in 1:L) tau[l] = 2.5 * tan(tau_unif[l]);
  B = Z * mu + (diag_pre_multiply(tau,L_Omega) * alpha)';
}

model {
  //priors
  to_vector(alpha) ~ normal(alpha_mean, alpha_scale);
  to_vector(mu) ~ cauchy(mu_mean, mu_scale);
  L_Omega ~ lkj_corr_cholesky(lkj_param);


  // model fitting
  for (r in 1:R) {
    for (t in 1:T) {
      Y[r, t] ~ categorical_logit(X[r, t]*B[r]');
    }
  }
}

generated quantities {
  // Yp is predicted choices for new data.
  int<lower=0, upper=A> Yhat[R, T];

  for (r in 1:R) {
    for (t in 1:T) {
      Yhat[r, t] = categorical_logit_rng(X[r, t]*B[r]');
    }
  }
}
