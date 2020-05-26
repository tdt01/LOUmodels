// Model 1a: Eigenvalues can only be real

data {
	int N;					// Sample size
	int Nsub;                               // Number of subjects

	int K;					// Number of items
	int R;					// Number of latent factors
	int p;					// Number of covariates

	int ID[N];

	int cumu[Nsub];
	int repme[Nsub];

	int Y[N, K];
	int missing_ID[N, K];

	vector[N] deltat;

	//vector[p] X[N];
	matrix[N, p] X;

	int ncate4;
	int ncate5;
	int ncate6;
	int ncate7;
}

parameters {

	real theta1;
	real theta2;
	real theta3;

	ordered[ncate4-1] theta4;
	ordered[ncate5-1] theta5;
	ordered[ncate6-1] theta6;
	ordered[ncate7-1] theta7;

	real mu_theta;
	real<lower=0.000001> sigma_theta;

	vector<lower=0.000001>[K] lambda;
	real<lower=0.000001> sigma_lambda;

	matrix[K, p] beta;

	matrix[Nsub, K] b_raw;
	vector<lower=0.000001>[K] sigma_bk;

	vector[R] xi[N];
	

	matrix[R, R] Gamma;

	real<lower=-1, upper=1> rho; // correlation coefficient
}

transformed parameters {

	matrix[Nsub, K] b;

	real<lower=0.000001> constraint1;
	real<lower=0.000001> constraint2;

	real<lower=0.0>      constraint3;

	corr_matrix[R] Omega;
	cov_matrix[R] Sigma;

	constraint1 = Gamma[1, 1] + Gamma[2, 2];
	constraint2 = Gamma[1, 1] * Gamma[2, 2] - Gamma[1, 2] * Gamma[2, 1];

	constraint3 = (Gamma[1, 1] - Gamma[2, 2])^2 + 4 * Gamma[1, 2] * Gamma[2, 1];

	Omega = [[1, rho], [rho, 1]];
	Sigma = Gamma * Omega + Omega * Gamma';

	for (i in 1 : Nsub){
		for (k in 1 : K){
			b[i, k] = b_raw[i, k] * sigma_bk[k];
		}
	}
}

model{
	// Prior

	theta1 ~ normal(mu_theta, sigma_theta);
	theta2 ~ normal(mu_theta, sigma_theta);
	theta3 ~ normal(mu_theta, sigma_theta);

	theta4 ~ normal(mu_theta, sigma_theta);
	theta5 ~ normal(mu_theta, sigma_theta);
	theta6 ~ normal(mu_theta, sigma_theta);
	theta7 ~ normal(mu_theta, sigma_theta);

	mu_theta ~ normal(0, 10);
	sigma_theta ~ cauchy(0, 5);

	lambda ~ normal(1, sigma_lambda);
	sigma_lambda ~ cauchy(0, 5);

	to_vector(beta) ~ cauchy(0, 5);

	sigma_bk ~ cauchy(0, 5);
	to_vector(b_raw) ~ normal(0, 1);

	to_vector(Gamma) ~ normal(0, 10);

	// At time=1

	for (i in 1 : Nsub){
		
		int k;

		k = cumu[i] - repme[i] + 1;

		xi[k] ~ multi_normal([0, 0]', Omega);
	}

	// Now is time = 2 to end

	for (i in 1 : Nsub){
		for (j in 2 : repme[i]){

			int k;

			matrix[R, R] Cova_trans;

			k = cumu[i] - repme[i] + j;
					
			Cova_trans = Omega - matrix_exp(-deltat[k] * Gamma) * Omega * matrix_exp(-deltat[k] * Gamma');
		
			xi[k] ~ multi_normal(matrix_exp(-deltat[k] * Gamma) * xi[k-1], Cova_trans);
		}
	}

	// likelihood

	for (i in 1 : N){

		if (missing_ID[i, 1] == 0){Y[i, 1] ~ bernoulli_logit(theta1 + beta[1, ] * X[i, ]' + lambda[1] * xi[i, 1] + b[ID[i], 1]);}
		if (missing_ID[i, 2] == 0){Y[i, 2] ~ bernoulli_logit(theta2 + beta[2, ] * X[i, ]' + lambda[2] * xi[i, 1] + b[ID[i], 2]);}
		if (missing_ID[i, 3] == 0){Y[i, 3] ~ bernoulli_logit(theta3 + beta[3, ] * X[i, ]' + lambda[3] * xi[i, 1] + b[ID[i], 3]);}

		if (missing_ID[i, 4] == 0){Y[i, 4] ~ ordered_logistic(beta[4, ] * X[i, ]' + lambda[4] * xi[i, 2] + b[ID[i], 4], theta4);}
		if (missing_ID[i, 5] == 0){Y[i, 5] ~ ordered_logistic(beta[5, ] * X[i, ]' + lambda[5] * xi[i, 2] + b[ID[i], 5], theta5);}
		if (missing_ID[i, 6] == 0){Y[i, 6] ~ ordered_logistic(beta[6, ] * X[i, ]' + lambda[6] * xi[i, 2] + b[ID[i], 6], theta6);}
		if (missing_ID[i, 7] == 0){Y[i, 7] ~ ordered_logistic(beta[7, ] * X[i, ]' + lambda[7] * xi[i, 2] + b[ID[i], 7], theta7);}
	}
}

generated quantities {


	matrix[R, R] A05;
	matrix[R, R] Cova_trans05;


	matrix[R, R] A10;
	matrix[R, R] Cova_trans10;


	matrix[R, R] A15;
	matrix[R, R] Cova_trans15;


	A05 = matrix_exp(-0.5 * Gamma);
	Cova_trans05 = Omega - matrix_exp(-0.5 * Gamma) * Omega * matrix_exp(-0.5 * Gamma');

	A10 = matrix_exp(- Gamma);
	Cova_trans10 = Omega - matrix_exp(- Gamma) * Omega * matrix_exp(- Gamma');

	A15 = matrix_exp(-1.5 * Gamma);
	Cova_trans15 = Omega - matrix_exp(-1.5 * Gamma) * Omega * matrix_exp(-1.5 * Gamma');
}




