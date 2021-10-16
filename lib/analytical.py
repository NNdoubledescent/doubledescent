import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm
from lib.utils import get_data, get_RQ


def analytical_results(
    seeds, n, d, p, k, noise, l2,
        fast_features_only=False, slow_features_only=False, n_epochs=1e4):

    s1 = 0.5
    s2 = 0.5 / k

    ts = 10 ** np.linspace(-3, 8, 300)
    lambdas = 1.0 / ts + l2

    # H_1 = p / d
    alpha_1 = n / p
    lambdas_1 = lambdas / ((p / d) * s1 ** 2)
    a1 = 1 + 2 * lambdas_1 / (1 - alpha_1 - lambdas_1 +
                              np.sqrt((1 - alpha_1 - lambdas_1) ** 2 +
                                      4 * lambdas_1))

    # H_2 = (d - p) / d
    alpha_2 = n / (d - p)
    lambdas_2 = lambdas / (((d - p) / d) * s2 ** 2)
    a2 = 1 + 2 * lambdas_2 / (1 - alpha_2 - lambdas_2 +
                              np.sqrt((1 - alpha_2 - lambdas_2) ** 2 +
                                      4 * lambdas_2))

    # assuming that the fast feature is already learned
    if slow_features_only:
        a1[:] = a1[-1]
    if fast_features_only:
        a2[:] = a2[0]

    R1 = (n / d) * 1 / a1
    R2 = (n / d) * 1 / a2

    b1 = alpha_1 / (a1 ** 2 - alpha_1)
    c1 = 1 + noise ** 2 - 2 * R2 - ((2 - a1) / a1) * (n / d)

    b2 = alpha_2 / (a2 ** 2 - alpha_2)
    c2 = 1 + noise ** 2 - 2 * R1 - ((2 - a2) / a2) * (n / d)

    Q1 = (b1 * b2 * c2 + b1 * c1) / (1 - b1 * b2)
    Q2 = (b1 * c1 * b2 + b2 * c2) / (1 - b1 * b2)

    R = R1 + R2
    Q = Q1 + Q2
    EG = 0.5 * (1 + noise ** 2 - 2 * R + Q)

    return ts + 1, R, Q, EG


def analytical_results_general_case(seeds, n, d, p, k, noise, l2, n_epochs=1e4):

    Rs = np.zeros((seeds, 300))
    Qs = np.zeros((seeds, 300))
    EG = np.zeros((seeds, 300))

    for seed in tqdm(range(seeds)):
        X, y, X_test, y_test, F, w = get_data(seed, n, d, p, k, noise)
        w_hat = np.zeros((d, 1))

        # eigendecomposition of the input covariance matrix
        XTX = np.dot(X.T, X)
        V, L, _ = np.linalg.svd(XTX)
        # optimal learning rates
        lr = 1.0 / L[0]

        # getting w_star using ridge
        w_star = np.zeros((d, 1))
        clf = Ridge(alpha=l2, fit_intercept=False)
        clf.fit(X, y[:, 0])
        w_star[:, 0] = clf.coef_.T

        ts = []
        for i, j in enumerate(np.linspace(-3, 8, 300)):
            t = (10 ** j)
            ts += [t + 1]

            # Gradient Descent
            w_hat = np.dot(V, np.dot(
                np.eye(d) - np.diag(np.exp(-lr * L * t)),
                np.dot(V.T, w_star)))
            # Uncomment for Gradient Flow instead
            # w_hat = np.dot(V, np.dot(
            #     np.eye(d) - np.diag(np.exp(-lr * L * t)),
            #     np.dot(V.T, w_star)))

            R, Q = get_RQ(w_hat, F, w, d)
            Rs[seed, i] = R
            Qs[seed, i] = Q
            EG[seed, i] = 0.5 * (1 - 2 * R + Q)

    return ts, Rs.mean(0), Qs.mean(0), EG.mean(0)
