data {
  int<lower=0> N;              // number of observations
  array[N] int<lower=0> y;     // response variable
  vector[N] x;                 // predictor variable
}

parameters {
  real beta;                   // intercept for p
  real beta1;                  // linear coefficient for p
  real beta2;                  // quadratic coefficient for p
  real beta3;                  // cubic coefficient for p

  real gamma;                  // intercept for pi
  real gamma1;                 // linear coefficient for pi
  real gamma2;                 // linear coefficient for pi

  vector<lower=0, upper=1>[N] p;  // individual probabilities
}

transformed parameters {
  vector[N] mu;                // mean parameter for beta distribution
  vector[N] pi;                // mixture probability (prob of binomial component)

  // Compute mean on probability scale using polynomial in x
  for (i in 1:N) {
    mu[i] = inv_logit(beta + beta1*x[i] + beta2*x[i]^2 + beta3*x[i]^3);
    pi[i] = inv_logit(gamma + gamma1*x[i] + gamma2*x[i]*x[i]);
  }
}

model {
  // Priors for p parameters
  beta ~ normal(0, 2);
  beta1 ~ normal(0, 2);
  beta2 ~ normal(0, 2);
  beta3 ~ normal(0, 2);

  // Priors for pi parameters
  gamma ~ normal(0, 2);
  gamma1 ~ normal(0, 40);
  gamma2 ~ normal(0, 40);

  // Beta prior for p using proportion parameterization
  p ~ beta_proportion(mu, 20); // control smoothing

  // Mixture likelihood
  for (i in 1:N) {
    if (y[i] == 15) {
      // If y[i] = 15, could come from either component
      target += log_sum_exp(
        log(pi[i]) + binomial_lpmf(y[i] | 15, p[i]),
        log1m(pi[i])
      );
    } else {
      // If y[i] != 15, must come from binomial component
      target += log(pi[i]) + binomial_lpmf(y[i] | 15, p[i]);
    }
  }
}