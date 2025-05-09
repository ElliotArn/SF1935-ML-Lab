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

# prior distribution for w0 & w1
mu = [0, 0]
alpha = 2
cov_matrix = (1 / alpha) * np.identity(2)

w0_linspace = np.linspace(-5, 5, 100)
w1_linspace = np.linspace(-5, 5 , 100)

w0arr, w1arr = np.meshgrid(w0_linspace, w1_linspace)
pos = np.dstack((w0arr, w1arr))

rv = multivariate_normal(mu, cov_matrix)
wpriorpdf = rv.pdf(pos)

plt.contour(w0arr, w1arr, wpriorpdf)
plt.show()

trn_data_size = 3

x_trn = np.linspace(-1, 1, 200)
x_trn_sample = np.random.choice(x_trn, trn_data_size, False)
t_trn_sample = w0_true + w1_true*x_trn_sample + np.random.normal(mean, sigma)

x_tst = np.concatenate(np.linspace(-1.5, -1.1, 5), np.linspace(1.1, 1.5, 5))
t_tst = w0_true + w1_true*x_tst + np.random.normal(mean, sigma)

w_transponent = np.transpose(wpriorpdf)
for sample in w_transponent:
    np.random.normal()






