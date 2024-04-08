# -*- coding: utf-8 -*-
import numpy as np


def get_interval(y, forecast_horizon, n_windows, coverage, model_obj, **kwargs):
    errors = build_dist(y,
                        forecast_horizon,
                        n_windows,
                        model_obj, **kwargs)
    horizon_errors = np.vstack(errors)
    fitted_errors = horizon_errors[:, 0]
    if isinstance(coverage, list):
        predicted_bounds = []
        fitted_bounds = []
        for bound in coverage:
            pred_bound = []
            for i in range(np.shape(errors)[1]):
                pred_bound.append(get_bound(horizon_errors[:, i], bound))
            fitted_bounds.append(get_bound(fitted_errors, bound))
            predicted_bounds.append(pred_bound)
    else:
        predicted_bounds = []
        for i in range(np.shape(errors)[1]):
            predicted_bounds.append(get_bound(horizon_errors[:, i], coverage))
        fitted_bounds = get_bound(fitted_errors, coverage)
    return fitted_bounds, predicted_bounds

def build_dist(y, forecast_horizon, n_splits, model_obj, **kwargs):
    residuals = []
    X = None
    if 'X' in kwargs.keys():
        X = kwargs['X']
        kwargs.pop('X')
    for split in range(n_splits):
        train_y = y[:-(split + forecast_horizon)]
        test_y = y[len(train_y): len(train_y) + forecast_horizon]
        if X is not None:
            train_X = X[:-(split + forecast_horizon), :]
            test_X = X[len(train_y): len(train_y) + forecast_horizon, :]
            model_obj.fit(train_y, X=train_X, **kwargs)
            prediction = model_obj.predict(forecast_horizon, X=test_X)
        else:
            model_obj.fit(train_y, **kwargs)
            prediction = model_obj.predict(forecast_horizon)
        residuals.append(test_y - prediction)
    return residuals

def get_bound(calibration_set, interval):
    return np.quantile(np.abs(calibration_set), interval)

def coverage_score(actual, upper, lower):
    scores = []
    for i in range(len(actual)):
        scores.append(actual[i] > upper[i] or actual[i] < lower[i])
    return 1 - sum(scores) / len(scores)
