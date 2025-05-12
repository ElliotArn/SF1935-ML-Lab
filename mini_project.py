import pylab as pb
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

np.random.seed(1)

# true parameters and error values
w0 = 0
w1 = 2.5
w2 = -0.5
err_mean = 0
err_sigma = 1.2

# Step 1 & 2
lin_size = 41
lin_trn_size = 13
lin_tst_size = 28

x1_lin = np.linspace(-1, 1, lin_size)
x2_lin = np.linspace(-1, 1, lin_size)

ti = np.zeros((lin_size, lin_size))
t_trn = np.zeros((lin_trn_size, lin_trn_size))
t_tst = np.zeros((lin_tst_size, lin_tst_size))

x1_tst_idx = 0
x2_tst_idx = 0
x1_trn_idx = 0
x2_trn_idx = 0

for x1_i in range(len(x1_lin)):
    print("h")
    x1 = x1_lin[x1_i]

    for x2_i in range(len(x1_lin)):
        x2 = x2_lin[x2_i]
        print(x1, x2)
        t_i = w0 + w1 * (x1 ** 2) + w2 * (x2 ** 3) + np.random.normal(err_mean, err_sigma)
        ti[x1_i][x2_i] = t_i

        if np.abs(x1) < 0.31 and np.abs(x2) < 0.31:
            print(x2_tst_idx)
            t_trn[x1_trn_idx][x2_trn_idx] = t_i
            x2_trn_idx += 1
        else:
            t_tst[x1_tst_idx][x2_tst_idx] = t_i + np.random.normal(err_mean, err_sigma)
            x2_tst_idx += 1

    x2_tst_idx = 0
    x2_trn_idx = 0

    if x1_i < 13:
        x1_trn_idx += 1

    if x1_i < 28:
        x1_tst_idx += 1

x1arr, x2arr = np.meshgrid(x1_lin, x2_lin)

plt.contourf(x1arr, x2arr, ti)
plt.title("x1 & x2 input space")
plt.xlabel("x1")
plt.xlim([-1, 1])
plt.ylabel("x2")
plt.ylim([-1, 1])
plt.show()

# Step 3

x1_lin_trn = np.linspace(-0.3, 0.3, lin_trn_size)
x2_lin_trn = np.linspace(-0.3, 0.3, lin_trn_size)

x1_lin_trn_pow2 = x1_lin_trn ** 2
x2_lin_trn_pow3 = x2_lin_trn ** 3

phi = np.vstack([np.ones(lin_trn_size), x1_lin_trn_pow2, x2_lin_trn_pow3])

w_ML = np.linalg.inv(phi.T @ phi) @ phi.T @ t_trn

sum_ML = 0

for t in range(lin_trn_size):
    sum_ML += (t_trn[t] - w_ML.T @ phi[t]) ** 2

beta_ML = 1 / ((1 / lin_trn_size) * sum_ML)

sum_MSE = 0

x1_lin_tst = np.linspace(-1, 0.35, lin_tst_size)
x2_lin_tst = np.linspace(-0.3, 0.3, lin_tst_size)


