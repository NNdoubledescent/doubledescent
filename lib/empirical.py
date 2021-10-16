import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm
from lib.utils import get_data, get_RQ


def ridge_regression(seeds, n, d, p, k, noise, l2):

    Rs = np.zeros((seeds, 50))
    Qs = np.zeros((seeds, 50))
    # to store the values of empirical generalization error
    EG_emp = np.zeros((seeds, 50))

    for seed in tqdm(range(seeds)):
        X, y, X_test, y_test, F, w = get_data(seed, n, d, p, k, noise)
        w_hat = np.zeros((d, 1))

        ts = []
        for i, j in enumerate(np.linspace(-3, 5, 50)):
            t = (10 ** j)
            lambda_ = (1.0 / t + l2)
            ts += [t + 1]

            clf = Ridge(alpha=lambda_, fit_intercept=False)
            clf.fit(X, y[:, 0])
            w_hat[:, 0] = clf.coef_.T

            R, Q = get_RQ(w_hat, F, w, d)
            Rs[seed, i] = R
            Qs[seed, i] = Q
            EG_emp[seed, i] = 0.5 * ((np.dot(X_test, w_hat) - y_test) ** 2).mean()

    return ts, Rs.mean(0), Qs.mean(0), EG_emp.mean(0)


def gradient_descent_np(seeds, n, d, p, k, noise, l2):

    Rs = np.zeros((seeds, 50))
    Qs = np.zeros((seeds, 50))
    # to store the values of empirical generalization error
    EG_emp = np.zeros((seeds, 50))

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
        for i, j in enumerate(np.linspace(-3, 5, 50)):
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
            EG_emp[seed, i] = 0.5 * ((np.dot(X_test, w_hat) - y_test) ** 2).mean()

    return ts, Rs.mean(0), Qs.mean(0), [EG_emp.mean(0), EG_emp.std(0)]


def gradient_descent_torch(seeds, n, d, p, k, noise, l2):
    Rs = np.zeros((seeds, 50))
    Qs = np.zeros((seeds, 50))
    # to store the values of empirical generalization error
    EG_emp = np.zeros((seeds, 50))

    for seed in tqdm(range(seeds)):
        X, y, X_test, y_test, F, w = get_data(seed, n, d, p, k, noise)
        w_hat = np.zeros((d, 1))

        # eigendecomposition of the input covariance matrix
        XTX = np.dot(X.T, X)
        V, L, _ = np.linalg.svd(XTX)
        # optimal learning rates
        lr = 1.0 / L[0]

        # simply creates a list of times at which we evaluate the model
        ts_ = []
        ts = (10 ** np.linspace(-3, 5, 50)).astype('int')
        last_t = -1
        for i, t in enumerate(ts):
            if t <= last_t:
                ts[i] = last_t + 1
            last_t = ts[i]
        # end of the simple part :)

        # training starts
        i = 0
        w_hat = np.zeros((d, 1))
        for epoch in tqdm(range(int(10 ** 5))):

            if epoch in ts:
                R, Q = get_RQ(w_hat, F, w, d)
                Rs[seed, i] = R
                Qs[seed, i] = Q
                EG_emp[seed, i] = 0.5 * ((np.dot(X_test, w_hat) - y_test) ** 2).mean()
                i += 1
                ts_ += [epoch + 1]

            # gradient descent update
            w_hat = w_hat - lr.item() * (np.dot(X.T, np.dot(X, w_hat) - y) + l2 * w_hat)

    return ts, Rs.mean(0), Qs.mean(0), EG_emp.mean(0)
