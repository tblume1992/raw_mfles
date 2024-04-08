# -*- coding: utf-8 -*-
from numba import jit, njit, vectorize
import numpy as np
from MFLES.utils import cap_outliers


@njit
def fsign(f):
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0

@njit
def soft_threshold(rho, alpha):
    """Soft threshold function used for Lasso regression"""
    if rho < -alpha:
        return rho + alpha
    elif rho > alpha:
        return rho - alpha
    else:
        return 0

# @njit
# def adaptive_lasso_coordinate_descent_step(X, y, w, alpha, j):
#     """Perform one coordinate descent step for Adaptive Lasso regression"""
#     n_samples = X.shape[0]
#     Xj = X[:, j]
#     Xj_dot = np.dot(Xj, Xj)
#     Xj_residual = Xj_dot * w[j] + np.dot(Xj, y - np.dot(X, w) + w[j] * Xj)
    
#     # Calculate adaptive weight based on the absolute value of the current parameter estimate
#     adaptive_weight = 1.0 / (np.abs(w[j]) + 1e-10)
    
#     w[j] = adaptive_weight * soft_threshold(Xj_residual, alpha) / Xj_dot
#     return w

@jit
def lasso_nb(X, y, alpha, tol=0.001, maxiter=10000):
    n, p = X.shape
    beta = np.zeros(p)
    R = y.copy()
    norm_cols_X = (X ** 2).sum(axis=0)
    resids = []
    prev_cost = 10e10
    for n_iter in range(maxiter):
        for ii in range(p):
            beta_ii = beta[ii]
            if beta_ii != 0.:
                R += X[:, ii] * beta_ii
            tmp = np.dot(X[:, ii], R)
            beta[ii] = fsign(tmp) * max(abs(tmp) - alpha, 0) / (.00001 + norm_cols_X[ii])
            if beta[ii] != 0.:
                R -= X[:, ii] * beta[ii]
        cost = (np.sum((y - X @ beta)**2) + alpha * np.sum(np.abs(beta))) / n
        resids.append(cost)
        if prev_cost - cost < tol:
            break
        else:
            prev_cost = cost
    return beta

# def adaptive_lasso_cd(X, y, alpha, tol=0.001, maxiter=10000):
#     n, p = X.shape
#     beta = np.zeros(p)
#     w = np.ones(p)  # Initialize weights for adaptive LASSO
#     R = y.copy()
#     norm_cols_X = (X ** 2).sum(axis=0)
#     resids = []
#     prev_cost = 10e10
#     for n_iter in range(maxiter):
#         for ii in range(p):
#             beta_ii = beta[ii]
#             if beta_ii != 0.:
#                 R += X[:, ii] * beta_ii
#             tmp = np.dot(X[:, ii], R)
#             z = np.abs(tmp)
#             w[ii] = 1.0 / (z + 1e-10)  # Update weights based on current estimate
#             beta[ii] = np.sign(tmp) * max(z - alpha, 0) / (.00001 + norm_cols_X[ii]) * w[ii]  # Use updated weights
#             if beta[ii] != 0.:
#                 R -= X[:, ii] * beta[ii]
#         cost = (np.sum((y - X @ beta)**2) + alpha * np.sum(np.abs(beta))) / n
#         resids.append(cost)
#         if prev_cost - cost < tol:
#             break
#         else:
#             prev_cost = cost
#     return beta

@vectorize
def calc_slope(x1,y1,x2,y2):
    xd = x2-x1
    if xd == 0:
        slope = 0
    else:
        slope = (y2-y1) / (xd)
    return slope

@njit
def siegel_repeated_medians(x,y):
    # Siegel repeated medians regression
    n_total = x.size
    slopes = np.empty((n_total), dtype=y.dtype)
    ints = np.empty((n_total), dtype=y.dtype)
    slopes_sub = np.empty((n_total-1), dtype=y.dtype)
    for i in range(n_total):
        for j in range(n_total):            
            if i == j:
                continue
            slopes_sub[j] = calc_slope(x[i],y[i],x[j],y[j])
        slopes[i] = np.median(slopes_sub)
        ints[i] = y[i] - slopes[i]*x[i]
    trend = x * np.median(slopes) + np.median(ints)
    return trend

@njit
def ses(y, alpha):
    results = np.zeros(len(y))
    results[0] = y[0]
    for i in range(1, len(y)):
        results[i] = alpha * y[i] + (1 - alpha) * results[i - 1]
    return results

@njit
def ses_ensemble(y, min_alpha=.05, max_alpha=1, smooth=0, order=1):
    #bad name but does either a ses ensemble or simple moving average
    results = np.zeros(len(y))
    iters = np.arange(min_alpha, max_alpha, .05)
    if smooth:
        for alpha in iters:
            results += ses(y, alpha)
        results = results / len(iters)
    else:
        results[:order] = y[:order]
        for i in range(1 + order, len(y)):
            results[i] += np.sum(y[i-order:i+1])/(order+1)
        results[:order + 1] = y[:order + 1] #fix this
    return results

@njit
def fast_ols(x, y):
    """Simple OLS for two data sets."""
    M = x.size

    x_sum = 0.
    y_sum = 0.
    x_sq_sum = 0.
    x_y_sum = 0.

    for i in range(M):
        x_sum += x[i]
        y_sum += y[i]
        x_sq_sum += x[i] ** 2
        x_y_sum += x[i] * y[i]

    slope = (M * x_y_sum - x_sum * y_sum) / (M * x_sq_sum - x_sum**2)
    intercept = (y_sum - slope * x_sum) / M

    return slope * x + intercept

@jit
def median(y, seasonal_period):
    n = len(y)
    if seasonal_period is None:
        return np.median(y) * np.ones(n)
    else:
        medians = np.zeros(n)
        for i in range(int(n / seasonal_period)):
            left = i * seasonal_period
            right = (1 + i) * seasonal_period
            medians[left: right] = np.median(y[left: right])
        remainder = n % seasonal_period
        if remainder:
            medians[right:] = np.median(y[left + remainder: ])
        return medians

@jit
def ols(X, y):
    coefs = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    return  np.sum(coefs * X, axis=1)

@jit
def wls(X, y, weights):
    weighted_X_T = X.T @ np.diag(weights)
    coefs = np.linalg.pinv(weighted_X_T.dot(X)).dot(weighted_X_T.dot(y))
    return  np.sum(coefs * X, axis=1)

@jit
def _ols(X, y):
    coefs = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    return  coefs

@jit
def ridge(X, y, lam):
    eye = np.eye(X.shape[1])
    coefs = np.linalg.pinv(X.T.dot(X) + eye@lam).dot(X.T.dot(y))
    return  np.sum(coefs * X, axis=1)

class OLS:
    def __init__(self):
        pass
    def fit(self, X, y):
        self.coefs = _ols(X, y)
    def predict(self, X):
        return  np.sum(self.coefs * X, axis=1)



