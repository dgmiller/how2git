data {
  int<lower=0> N; // number of observations
  vector[N] x; // input
  int<lower=0, upper=1> Y[N]; // outcome
}

parameters {
  real beta;
  real alpha;
}

model {
  Y ~ bernoulli_logit(alpha + beta * x);
}
