functions {
    vector deterministic_model(
        int N,
        matrix X,
        vector w
    ){
        vector[N] z;
        z = X * w;
        return z;
    }

    vector log_likelihood(
        int N,
        matrix X,
        vector y,
        vector w,
        real sigma
    ){
        vector[N] z;
        vector[N] logp;
        z = deterministic_model(N, X, w);
        for (n in 1:N){
            logp[n] = normal_lpdf(y[n] | z[n], sigma);
        }
        return logp;
    }
}

data {
    int<lower=0> N; // number of data items
    int<lower=0> D; // number of dimention
    matrix[N, D] X; // data matrix
    vector[N] y; // target vector
    int<lower=0> N_new; // number of new data items
    matrix[N_new, D] X_new; // new data matrix
}

parameters {
    vector[D] w; // weights for linear regression
    real<lower=0> sigma; // noise
}

transformed parameters {
    vector[N] logp;
    logp = log_likelihood(N, X, y, w, sigma);
}

model {
    w ~ normal(0, 1);
    sigma ~ cauchy(0, 1);
    for (n in 1:N){
        target += logp[n];
    }
}

generated quantities {
    array[N_new] real y_new;
    if (N_new > 0){
        y_new = normal_rng(deterministic_model(N_new, X_new, w), sigma);
    }
}