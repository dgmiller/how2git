// Data values, hyperparameters, observed choices, and the experimental design.
data {
  int<lower=1> R; // Number of respondents.
  int<lower=1> T; // Number of choice tasks per respondent.
  int<lower=2> A; // Number of product alternatives per choice task.
  int<lower=1> L; // Number of (estimable) attribute levels.
  int<lower=1> C; // Number of respondent-level covariates.
  

  matrix[A, L] X[R, T];          // Array of experimental designs per choice task.
  int<lower=1, upper=A> Y[R, T]; // Matrix of observed choices.
  matrix[R, C] Z;                // Matrix of respondent-level covariates.

  real mu_mean;           // Mean of coefficients for the heterogeneity model.
  real<lower=0> mu_scale; // Scale of coefficients for the heterogeneity model.
  real alpha_mean;             // Mean of scale parameters for the heterogeneity model.
  real<lower=0> alpha_scale;   // Scale of scale parameters for the heterogeneity model.
  real<lower=0> lkj_param; // Shape of correlation matrix for the heterogeneity model.
}

parameters {
  matrix[R, L] B;        // Matrix of beta (part-worth) coefficients.
  matrix[C, L] mu;       // Matrix of coefficients for the heterogeneity model.
  vector<lower = 0>[L] tau; // Vector of scale parameters for the heterogeneity model.
  corr_matrix[L] Omega;     // Correlation matrix for the heterogeneity model.
}

transformed parameters {
  // Covariance matrix for the heterogeneity model.
  cov_matrix[L] Sigma = quad_form_diag(Omega, tau);
  matrix[R, L] Zmu = Z * mu;
}

model {
  // Hyperpriors on mu, tau, and Omega (and thus Sigma).
  to_vector(mu) ~ normal(mu_mean, mu_scale);
  tau ~ cauchy(alpha_mean, alpha_scale);
  Omega ~ lkj_corr(lkj_param);
  
  // Hierarchical multinomial logit.
  for (r in 1:R) {
    B[r, ] ~ multi_normal(Z[r,] * mu, Sigma);    
    for (t in 1:T) {
      Y[r, t] ~ categorical_logit(X[r, t] * B[r,]');
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
