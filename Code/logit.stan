// conditional logit for DCEs
data {
  int<lower=2> A; // number of alternatives (choices) per question
  int<lower=1> L; // number of feature variables
  int<lower=1> N; // number of observations
  int<lower=1> Ntest; // number of out of sample observations

  matrix[A, L] X[N]; // matrix of attributes for each obs
  int<lower=1, upper=A> Y[N]; // observed responses

  matrix[A, L] Xhat[Ntest];

  real mu; // mean for Beta
  real<lower=0> sigma; // variance of Beta

}

parameters {
  vector[L] Beta; // matrix of beta coefficients
}

model {
  Beta ~ normal(mu, sigma); // prior for Beta
  for (n in 1:N) {
    Y[n] ~ categorical_logit(X[n]*Beta);
  }
}

generated quantities {
  // Yhat is predicted choices for new data.
  int<lower=0, upper=A> Yhat[Ntest];

  for (n in 1:Ntest) {
    Yhat[n] = categorical_logit_rng(Xhat[n]*Beta);
  }
}
