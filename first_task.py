import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

# true parameters and error values
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
cov_matrix = true_variance * np.identity(2)

linspace_size = 200
w0_linspace = np.linspace(-2, 2, linspace_size)
w1_linspace = np.linspace(-2, 2 , linspace_size)

w0arr, w1arr = np.meshgrid(w0_linspace, w1_linspace)
pos = np.dstack((w0arr, w1arr))
rv = multivariate_normal(mu, cov_matrix)
wpriorpdf = rv.pdf(pos)

plt.contour(w0arr, w1arr, wpriorpdf)
plt.title("Prior distribution space for w0 & w1")
plt.xlabel("w0")
plt.xlim([-2.5, 2.5])
plt.ylabel("w1")
plt.ylim([-2.5, 2.5])
plt.show()

# Step 2: plot likelihood across all w in parameter space for subset data
trn_data_size = 100

x_trn = np.linspace(-1, 1, 200)
x_trn_sample = np.random.choice(x_trn, trn_data_size, False)
t_trn_sample = w0_true + w1_true*x_trn_sample + np.random.normal(error_mean, error_sigma)

likelihood_grid = np.zeros((linspace_size, linspace_size))

# basically, what is the probability of finding the true t values from (t_trn_sample) in a normal distribution with
# mean = t_prediction over all the different normal distributions (all pairs of (w0, w1))?
for w0_index in range(linspace_size):
    w0_value = w0_linspace[w0_index]
    for w1_index in range(linspace_size):
        w1_value = w1_linspace[w1_index]

        t_prediction = w0_value + w1_value * x_trn_sample
        likelihood = np.prod(multivariate_normal.pdf(t_trn_sample, t_prediction, true_variance))

        likelihood_grid[w0_index, w1_index] = likelihood

plt.contour(w0arr, w1arr, likelihood_grid)
plt.title("Likelihood over w0 & w1 parameter space")
plt.xlabel("w0")
plt.xlim([0, 2])
plt.ylabel("w1")
plt.ylim([-2, 0])
plt.show()

# Step 3: posterior distribution for w0 & w1




# Step 4:

x_tst = np.concatenate(np.linspace(-1.5, -1.1, 5), np.linspace(1.1, 1.5, 5))
t_tst = w0_true + w1_true*x_tst + np.random.normal(error_mean, error_sigma)