# simple example taken from https://ecosang.github.io/blog/study/dirichlet-process-with-stan/

library(mixtools)
library(ggplot2)
library(tidyverse)
library(magrittr)
# Data generation code retrieved from
# http://www.jarad.me/615/2013/11/13/fitting-a-dirichlet-process-mixture

dat_generator <- function(truth) {
  set.seed(1)
  n = 500
  
  f = function(x) {
    out = numeric(length(x))
    for (i in 1:length(truth$pi)) out = out + truth$pi[i] * dnorm(x, truth$mu[i], 
                                                                  truth$sigma[i])
    out
  }
  y = rnormmix(n, truth$pi, truth$mu, truth$sigma)
  for (i in 1:length(truth$pi)) {
    assign(paste0("y", i), rnorm(n, truth$mu[i], truth$sigma[i]))
  }
  dat <- tibble(y = y, y1 = y1, y2 = y2, y3 = y3)
}
truth = data.frame(pi = c(0.1, 0.5, 0.4), mu = c(-3, 0, 3), sigma = sqrt(c(0.5, 
                                                                           0.75, 1)))
dat <- dat_generator(truth)

ggplot(data = dat %>% gather(key, value), aes(value)) + geom_density(aes(color = key)) + 
  theme_bw() + xlab("y") + ggtitle("y is mixture of {y1,y2,y3}")

library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_model <- "
data{
  int<lower=0> C;//num of cluster
  int<lower=0> N;//data num
  real y[N];
}

parameters {
  real mu_cl[C]; //cluster mean
  real <lower=0,upper=1> v[C];
  real<lower=0> sigma_cl[C]; // error scale
  //real<lower=0> alpha; // hyper prior DP(alpha,base)
}

transformed parameters{
  simplex [C] pi;
  pi[1] = v[1];
  // stick-break process based on The BUGS book Chapter 11 (p.294)
  for(j in 2:(C-1)){
      pi[j]= v[j]*(1-v[j-1])*pi[j-1]/v[j-1]; 
  }
  pi[C]=1-sum(pi[1:(C-1)]); // to make a simplex.
}

model {
  real alpha = 1;
  real a=0.001;
  real b=0.001;
  real ps[C];
  sigma_cl ~ inv_gamma(a,b);
  mu_cl ~ normal(0,5);
  //alpha~gamma(6,1);
  v ~ beta(1,alpha);
  
  for(i in 1:N){
    for(c in 1:C){
      ps[c]=log(pi[c])+normal_lpdf(y[i]|mu_cl[c],sigma_cl[c]);
    }
    target += log_sum_exp(ps);
  }

}
"
y <- dat$y
C <- 8  # to ensure large enough
N <- length(y)
input_dat <- list(y = y, N = N, C = C)
# model_object<-stan_model(model_code=stan_model)
fit <- stan(model_code = stan_model, data = input_dat, iter = 1000, chains = 1)
results <- rstan::extract(fit)


plot_dat_pi <- data.frame(results$pi) %>% as_data_frame() %>% set_names(sprintf("pi%02d", 
                                                                                1:8))

ggplot(data = plot_dat_pi %>% gather(key, value), aes(x = key, y = value)) + 
  geom_boxplot() + theme_bw()

library(gridExtra)
plot_mu_dat <- data.frame(results$mu_cl[, 1:3]) %>% as_data_frame() %>% set_names(sprintf("mu%d", 
                                                                                          1:3))
plot_mu_dat %<>% mutate(xgrid = (1:length(plot_mu_dat$mu1)))
ggplot(plot_mu_dat %>% gather(key, value, mu1:mu3), aes(x = xgrid, y = value, 
                                                        color = key)) + geom_point() + theme_bw()

knitr::kable(truth %>% as_data_frame())

mean_results <- data.frame(pi = colMeans(results$pi[, 1:3]), mu = colMeans(results$mu[, 
                                                                                      1:3]), sigma = colMeans(results$sigma[, 1:3]))
knitr::kable(mean_results)

# do we recover the same parameters using another (good) technique
library(mclust)
mc <- Mclust(y, G = 1:8)

d <- data.frame(y = y, fit = predict(mc, y)$classification)
d %>% group_by(fit) %>%
  summarise(m = mean(y),
            sigma = sd(y))

ggplot(d, aes(x = y, colour = factor(fit), group = factor(fit))) +
  geom_density() +
  geom_density(data = dat %>% gather(key, value), inherit.aes = FALSE, 
               aes(value, colour = key))
