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
err_sigma = 0.3

# Step 1 & 2
lin_size = 41

x1_lin = np.linspace(-1, 1, lin_size)
x2_lin = np.linspace(-1, 1, lin_size)

ti = np.zeros((lin_size, lin_size))
t_trn = np.zeros((lin_size, lin_size))
t_tst = np.zeros((lin_size, lin_size))

for x1_i in range(len(x1_lin)):
    x1 = x1_lin[x1_i]
    for x2_i in range(len(x1_lin)):
        x2 = x2_lin[x2_i]
        t_i = w0 + w1 * (x1 ** 2) + w2 * (x2 ** 3) + np.random.normal(err_mean, err_sigma)
        ti[x1_i][x2_i] = t_i

        if np.abs(x1) > 0.31 and np.abs(x2) > 0.31:
            t_tst[x1_i][x2_i] = t_i + np.random.normal(err_mean, err_sigma)
        else:
            t_trn[x1_i][x2_i] = t_i

x1arr, x2arr = np.meshgrid(x1_lin, x2_lin)

plt.contourf(x1arr, x2arr, ti)
plt.title("x1 & x2 input space")
plt.xlabel("x1")
plt.xlim([-1, 1])
plt.ylabel("x2")
plt.ylim([-1, 1])
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1arr, x2arr, ti.T, cmap='viridis')
ax.set_title("x1 & x2 with corresponding values of t")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("t")
plt.tight_layout()
plt.show()

# Step 3
trn_x1, trn_x2, trn_t = [], [], []

for i in range(lin_size):
    for j in range(lin_size):
        if t_trn[i, j] != 0:
            trn_x1.append(x1_lin[i])
            trn_x2.append(x2_lin[j])
            trn_t.append(t_trn[i, j])

trn_x1 = np.array(trn_x1)
trn_x2 = np.array(trn_x2)
trn_t = np.array(trn_t)

phi_trn = np.vstack([np.ones_like(trn_x1), trn_x1 ** 2, trn_x2 ** 3]).T

w_ML = np.linalg.inv(phi_trn.T @ phi_trn) @ phi_trn.T @ trn_t

MSE_ML = np.mean((phi_trn @ w_ML - trn_t) ** 2)
print(f"MSE_ML: {MSE_ML}")
beta_ML = 1 / MSE_ML

# Step 4
alpha = 0.3
beta = 1.0 / (err_sigma**2)
I = np.eye(phi_trn.shape[1])

SN_inv = alpha * I + beta * (phi_trn.T @ phi_trn)
SN = np.linalg.inv(SN_inv)
mN = beta * SN @ phi_trn.T @ trn_t

tst_x1, tst_x2, tst_t = [], [], []

for i in range(lin_size):
    for j in range(lin_size):
        if t_tst[i, j] != 0:
            tst_x1.append(x1_lin[i])
            tst_x2.append(x2_lin[j])
            tst_t.append(t_tst[i, j])

tst_x1 = np.array(tst_x1)
tst_x2 = np.array(tst_x2)
tst_t = np.array(tst_t)

phi_tst = np.vstack([np.ones_like(tst_x1), tst_x1 ** 2, tst_x2 ** 3]).T

mean_bayes_tst = phi_tst @ mN
varN_bayes_tst = np.array([(1 / beta) + phi @ SN @ phi.T for phi in phi_tst])
MSE_bayes_tst = np.mean((tst_t - mean_bayes_tst) ** 2)
print(f"MSE_bayes (test data): {MSE_bayes_tst}")

# Step 6
mean_bayes_trn = phi_trn @ mN
varN_bayes_trn = np.array([(1 / beta) + phi @ SN @ phi.T for phi in phi_trn])
MSE_bayes_trn = np.mean((trn_t - mean_bayes_trn) ** 2)
print(f"MSE_bayes (training data): {MSE_bayes_trn}")
