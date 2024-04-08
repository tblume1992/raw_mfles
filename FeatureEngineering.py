# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, njit
from MFLES.utils import fastsin,fastcos


@njit
def fourier(y, seasonal_period, fourier_order):
    n = len(y)
    m = fourier_order * 2
    results = np.zeros((n, m))
    for j in range(fourier_order):
        for i in range(n):
            x = (j + 1) * i
            results[i, j] = fastcos(x) / seasonal_period
            results[i, m-j-1] = fastsin(x) / seasonal_period
    return results

@njit
def get_fourier_series(length, seasonal_period, fourier_order):
    x = 2 * np.pi * np.arange(1, fourier_order + 1) / seasonal_period
    t = np.arange(1, length + 1).reshape(-1, 1)
    x = x * t
    fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return fourier_series

@njit
def get_basis(y, n_changepoints, decay=-1, gradient_strategy=0):
    y = y.copy()
    y -= y[0]
    n = len(y)
    if gradient_strategy:
        gradients = np.abs(y[:-1] - y[1:])
    initial_point = y[0]
    final_point = y[-1]
    mean_y = np.mean(y)
    changepoints = np.zeros(shape=(len(y), n_changepoints + 1))
    array_splits = []
    for i in range(1, n_changepoints + 1):
        i = n_changepoints - i + 1
        if gradient_strategy:
            cps = np.argsort(gradients)[::-1]
            cps = cps[cps > .1 * len(gradients)]
            cps = cps[cps < .9 * len(gradients)]
            split_point = cps[i-1]
            array_splits.append(y[:split_point])
        else:
            split_point = len(y)//i
            array_splits.append(y[:split_point])
            y = y[split_point:]
    len_splits = 0
    for i in range(n_changepoints):
        if gradient_strategy:
            len_splits = len(array_splits[i])
        else:
            len_splits += len(array_splits[i])
        moving_point = array_splits[i][-1]
        left_basis = np.linspace(initial_point,
                                  moving_point,
                                  len_splits)
        if decay is None:
            end_point = final_point
        else:
            if decay == -1:
                dd = moving_point**2 / (mean_y**2)
                if dd > .99:
                    dd = .99
                if dd < .001:
                    dd = .001
                end_point = moving_point - ((moving_point - final_point) * (1 - dd))
            else:
                end_point = moving_point - ((moving_point - final_point) * (1 - decay))
        right_basis = np.linspace(moving_point,
                                  end_point,
                                  n - len_splits + 1)
        changepoints[:, i] = np.append(left_basis, right_basis[1:])
    changepoints[:, i+1] = np.ones(n)
    return changepoints

@jit
def get_future_basis(basis_functions, forecast_horizon):
    n_components = np.shape(basis_functions)[1]
    slopes = np.gradient(basis_functions)[0][-1, :]
    future_basis = np.array(np.arange(0, forecast_horizon + 1))
    future_basis += len(basis_functions)
    future_basis = np.transpose([future_basis] * n_components)
    future_basis = future_basis * slopes
    future_basis = future_basis + (basis_functions[-1, :] - future_basis[0, :])
    return future_basis[1:, :]