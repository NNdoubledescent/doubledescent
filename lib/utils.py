import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')
from scipy.stats import ortho_group
mpl.use('tkagg')
np.random.seed(1234)


# Implements the teacher and generates the data
def get_data(seed, n, d, p, k, noise):
    np.random.seed(seed)
    Z = np.random.randn(n, d) / np.sqrt(d)
    Z_test = np.random.randn(1000, d) / np.sqrt(d)

    # teacher
    w = np.random.randn(d, 1)
    y = np.dot(Z, w)
    y = y + noise * np.random.randn(*y.shape)
    # test data is noiseless
    y_test = np.dot(Z_test, w)

    # the modulation matrix that controls students access to the data
    F = get_modulation_matrix(d, p, k)

    # X = F^T Z
    X = np.dot(Z, F)
    X_test = np.dot(Z_test, F)

    return X, y, X_test, y_test, F, w


def get_RQ(w_hat, F, w, d):
    # R: the alignment between the teacher and the student
    R = np.dot(np.dot(F, w_hat).T, w).item() / d
    # Q: the student's modulated norm
    Q = np.dot(np.dot(F, w_hat).T, np.dot(F, w_hat)).item() / d
    return R, Q


def get_optimal_lr(X):
    XTX = np.dot(X.T, X)
    V, L, _ = torch.svd(XTX)
    return 1.0 / L[0]


def get_modulation_matrix(d, p, k):
    U = ortho_group.rvs(d)
    VT = ortho_group.rvs(d)
    S = np.eye(d)
    S[:p, :p] *= 1
    S[p:, p:] *= 1 / k
    F = np.dot(U, np.dot(S, VT))
    return F
