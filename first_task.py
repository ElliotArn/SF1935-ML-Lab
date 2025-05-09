import pylab as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist

w0_true = -1.2
w1_true = 0.9

my = 0
variance = 0.2
sigma = np.sqrt(variance)

x_trn = np.linspace(-1,1,200)

x_tst = np.concatenate(np.linspace(-1.5, -1.1, 5), np.linspace(1.1, 1.5, 5))
