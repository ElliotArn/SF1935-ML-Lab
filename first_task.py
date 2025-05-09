import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

w0_true = -1.2
w1_true = 0.9
mean = 0
variance = 0.2
sigma = np.sqrt(variance)

# Step 1: prior distribution for w0 & w1
mu = [0, 0]
alpha = 0.5
cov_matrix = (1 / alpha) * np.identity(2)

linspace_size = 400
w0_linspace = np.linspace(-5, 5, linspace_size)
w1_linspace = np.linspace(-5, 5 , linspace_size)

w0arr, w1arr = np.meshgrid(w0_linspace, w1_linspace)
pos = np.dstack((w0arr, w1arr))

rv = multivariate_normal(mu, cov_matrix)
wpriorpdf = rv.pdf(pos)

plt.contour(w0arr, w1arr, wpriorpdf)
plt.show()

# Step 2: plot likelihood across all w in parameter space for subset data
trn_data_size = 10

x_trn = np.linspace(-1, 1, 200)
x_trn_sample = np.random.choice(x_trn, trn_data_size, False)
t_trn_sample = w0_true + w1_true*x_trn_sample + np.random.normal(mean, sigma)

#x_tst = np.concatenate(np.linspace(-1.5, -1.1, 5), np.linspace(1.1, 1.5, 5))
#t_tst = w0_true + w1_true*x_tst + np.random.normal(mean, sigma)

likelihood_grid = np.zeros((linspace_size, linspace_size))

# basically, what is the probability of finding the true t value (t_trn_sample) in a normal distribution with mean
# t_prediction over all the different normal distributions (all pairs of (w0, w1))?
for w0_index in range(linspace_size):
    for w1_index in range(linspace_size):
        w0_value = w0_linspace[w0_index]
        w1_value = w1_linspace[w1_index]

        t_prediction = w0_value + w1_value * x_trn_sample
        likelihood = np.prod(multivariate_normal.pdf(t_trn_sample, t_prediction, variance))

        likelihood_grid[w0_index, w1_index] = likelihood

plt.contour(w0arr, w1arr, likelihood_grid)
plt.show()
