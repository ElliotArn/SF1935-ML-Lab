import pylab as pb
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

# true parameters and error values
np.random.seed(16)
w0_true = -1.2
w1_true = 0.9
error_mean = 0
error_variance = 0.2
error_sigma = np.sqrt(error_variance)

# Step 1: prior distribution for w0 & w1
mu = [0, 0]
alpha = 2
true_variance = 1 / alpha
true_sigma = np.sqrt(true_variance)
cov_matrix_prior = true_variance * np.identity(2)

linspace_size = 200
w0_linspace = np.linspace(-5, 5, linspace_size)
w1_linspace = np.linspace(-5, 5 , linspace_size)

w0arr, w1arr = np.meshgrid(w0_linspace, w1_linspace)
pos = np.dstack((w0arr, w1arr))
rv_prior = multivariate_normal(mu, cov_matrix_prior)
wpriorpdf = rv_prior.pdf(pos)

plt.subplot(3, 1, 1)
plt.contour(w0arr, w1arr, wpriorpdf)
plt.title("Prior distribution space for w0 & w1")
plt.xlabel("w0")
plt.xlim([-5, 5])
plt.ylabel("w1")
plt.ylim([-5, 5])

# Step 2: plot likelihood across all w in parameter space for subset data
trn_data_size = 50

x_trn = np.linspace(-1, 1, 200)
x_trn_sample = np.random.choice(x_trn, trn_data_size, False)
t_trn_sample = w0_true + w1_true*x_trn_sample + np.random.normal(error_mean, error_sigma, size = trn_data_size)

likelihood_grid = np.zeros((linspace_size, linspace_size))
t_prediction = 0

# basically, what is the probability of finding the true t values from (t_trn_sample) in a normal distribution with
# mean = t_prediction over all the different normal distributions (all pairs of (w0, w1))?
for w0_index in range(linspace_size):
    w0_value = w0_linspace[w0_index]
    for w1_index in range(linspace_size):
        w1_value = w1_linspace[w1_index]

        t_prediction = w0_value + w1_value * x_trn_sample
        likelihood = np.prod(multivariate_normal.pdf(t_trn_sample, t_prediction, true_variance))

        likelihood_grid[w0_index, w1_index] = likelihood

plt.subplot(3, 1, 2)
plt.contour(w0arr, w1arr, likelihood_grid)
plt.title("Likelihood over w0 & w1 parameter space")
plt.xlabel("w0")
plt.xlim([-5, 5])
plt.ylabel("w1")
plt.ylim([-5, 5])

# Step 3: posterior distribution for w0 & w1

beta = 1 / error_variance

x_ext = np.transpose(np.vstack((np.ones(trn_data_size), x_trn_sample)))
x_ext_tranpose = np.transpose(x_ext)

inv_cov_matrix_posterior = alpha * np.identity(2) + beta * x_ext_tranpose @ x_ext
cov_matrix_posterior = np.linalg.inv(inv_cov_matrix_posterior)

mean_posterior = beta * cov_matrix_posterior @ x_ext_tranpose @ t_trn_sample.reshape(-1, 1)
print(mean_posterior)

rv_post = multivariate_normal(mean_posterior.ravel(), cov_matrix_posterior)
wposteriorpdf = rv_post.pdf(pos)

plt.subplot(3, 1, 3)
plt.contour(w0arr, w1arr, wposteriorpdf)
plt.title("Posterior distribution space for w0 & w1")
plt.xlabel("w0")
plt.xlim([-5, 5])
plt.ylabel("w1")
plt.ylim([-5, 5])
plt.show()

# Step 4: Plot sampled regression lines from posterior

x_tst = np.concatenate((np.linspace(-1.5, -1.1, 5), np.linspace(1.1, 1.5, 5)))
t_tst = w0_true + w1_true * x_tst + np.random.normal(error_mean, error_sigma, size = len(x_tst))

wsamples = rv_post.rvs(size=5)

plt.subplot(2, 1, 1)
for w in wsamples:
    t_pred = w[0] + w[1] * x_trn
    plt.plot(x_trn, t_pred, label=f"w0={w[0]:.2f}, w1={w[1]:.2f}")

plt.scatter(x_trn_sample, t_trn_sample, color='blue', label="Training data")
plt.scatter(x_tst, t_tst, color='red', label="Test data")

plt.title("Sampled regression lines from posterior")
plt.xlabel("x")
plt.ylabel("t")
plt.xlim([-2, 2])
plt.ylim([-3, 0.5])
plt.grid(True)
plt.legend()

# Step 5: Make predictions on testing data using the Bayesian predictive distribution

phi_x_tst = np.transpose(np.vstack((np.ones_like(x_tst), x_tst)))

mean_preds = phi_x_tst @ mean_posterior

var_preds = []
for phi in phi_x_tst:
    var = (1 / beta) + phi.T @ cov_matrix_posterior @ phi
    var_preds.append(var)
var_preds = np.array(var_preds)
std_preds = np.sqrt(var_preds)

plt.subplot(2, 1, 2)
plt.errorbar(x_tst, mean_preds.ravel(), yerr=std_preds, fmt='o', color='green', label='Predictive mean ± std')
plt.scatter(x_trn_sample, t_trn_sample, color='blue', label='Training data')
plt.scatter(x_tst, t_tst, color='red', label='Test targets')


# part 6
max_like_w = np.linalg.inv(x_ext_tranpose @ x_ext) @ x_ext_tranpose @ t_trn_sample.reshape(-1,1)

max_like_x = np.linspace(-1.5, 1.5, 10)
max_like_x_ext = np.column_stack((np.ones_like(max_like_x), max_like_x))
max_like_y = max_like_x_ext @ max_like_w 
plt.plot(max_like_x, max_like_y, label='maxlike', color='orange')

print(max_like_w)

plt.title("Bayesian predictive distribution on test data")
plt.xlabel("x")
plt.ylabel("t")
plt.xlim([-2, 2])
plt.ylim([-3, 1])
plt.grid(True)
plt.legend()
plt.show()


