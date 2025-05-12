import pylab as pb
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

true_w = [0, 2.5, -0.5]
err_mean = 0
err_sigma = 0.3

np.set_printoptions(threshold=np.inf)

x1 = np.linspace(-1, 1, 41)
x2 = np.linspace(-1, 1, 41)
X1, X2 = np.meshgrid(x1, x2)

X = np.column_stack((X1.ravel(), X2.ravel()))

test_mask = (np.abs(X[:,0]) > 0.3) & (np.abs(X[:,1]) > 0.3)

X_test = X[test_mask]
X_train = X[~test_mask]
t_test = list(map(lambda pair: true_w[1] * (pair[0]**2) + true_w[2] * (pair[1]**3), X_test))
t_test = np.array(t_test)
t_test +=  np.random.normal(err_mean, err_sigma, size = t_test.shape)
t_test +=  np.random.normal(err_mean, err_sigma, size = t_test.shape)
t_train = list(map(lambda pair: true_w[1] * (pair[0]**2) + true_w[2] * (pair[1]**3), X_train))
t_train = np.array(t_train)
t_train += np.random.normal(err_mean, err_sigma, size = t_train.shape)

def phi(x):
    return np.column_stack((np.ones(x.shape[0]), x[:,0]**2, x[:,1]**3))

phi_train = phi(X_train)
phi_test = phi(X_test)
w_max_like = np.linalg.inv(phi_train.T @ phi_train) @ phi_train.T @ t_train
mse = np.mean((w_max_like @ phi_test.T - t_test) ** 2)
print(mse)

beta = 1/mse
alpha = [0.3, 0.7, 2.0]

sn1 = np.linalg.inv( alpha[0] * np.eye(3) + beta* phi_train.T @ phi_train)
sn2 = np.linalg.inv( alpha[1] * np.eye(3) + beta* phi_train.T @ phi_train)
sn3 = np.linalg.inv( alpha[2] * np.eye(3) + beta* phi_train.T @ phi_train)

mean1 = beta * sn1 @ phi_train.T @ t_train
mean2 = beta * sn2 @ phi_train.T @ t_train
mean3 = beta * sn3 @ phi_train.T @ t_train

def normal_dis_ans(mean, sn):
    list = []
    for x in phi_test:
        var = mse + x.T @ sn1 @ x
        list.append(np.random.normal(mean1 @ x, math.sqrt(var)))
    return list

normal_test1 = normal_dis_ans(mean1, sn1)
normal_test2 = normal_dis_ans(mean2, sn2)
normal_test3 = normal_dis_ans(mean3, sn3)

