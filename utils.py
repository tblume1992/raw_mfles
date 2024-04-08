import itertools
import numpy as np
from numba import jit, njit, vectorize

def cap_outliers(series, outlier_cap=3):
    mean = np.mean(series)
    std = np.std(series)
    series = series.clip(min=mean - outlier_cap * std,
                         max=mean + outlier_cap * std)
    return series

@njit
def set_fourier(period):
    if period < 10:
        fourier = 5
    elif period < 70:
        fourier = 10
    else:
        fourier = 15
    return fourier

@njit
def calc_trend_strength(resids, deseasonalized):
    return max(0, 1-(np.var(resids)/np.var(deseasonalized)))

@njit
def calc_seas_strength(resids, detrended):
    return max(0, 1-(np.var(resids)/np.var(detrended)))

@njit
def calc_rsq(y, fitted):
    try:
        mean_y = np.mean(y)
        ssres = 0
        sstot = 0
        for vals in zip(y, fitted):
            sstot += (vals[0] - mean_y) ** 2
            ssres += (vals[0] - vals[1]) ** 2
        return 1 - (ssres / sstot)
    except:
        return 0

@njit
def calc_cov(y, mult=1):
    if mult:
        # source http://medcraveonline.com/MOJPB/MOJPB-06-00200.pdf
        return np.sqrt(np.exp(np.log(10)*(np.std(y)**2) - 1))
    else:
        return np.std(y) / np.mean(y)


@njit
def fastsin(x):
    return np.sin(2* np.pi * x)


@njit
def fastcos(x):
    return np.cos(2 * np.pi * x)

@njit
def calc_mse(actual, fitted):
    mse = np.zeros(len(actual))
    for i, val in enumerate(zip(actual, fitted)):
        mse[i] = (val[0] - val[1])**2
    return np.mean(mse)

@njit
def calc_mae(actual, fitted):
    mae = np.zeros(len(actual))
    for i, val in enumerate(zip(actual, fitted)):
        mae[i] = np.abs(val[0] - val[1])
    return np.mean(mae)

@njit
def calc_mape(actual, fitted):
    eps = 0.000001
    mape = np.zeros(len(actual))
    for i, val in enumerate(zip(actual, fitted)):
        mape[i] = np.abs(val[0] - val[1]) / (eps + val[0])
    return np.mean(mape)

@njit
def calc_smape(actual, fitted):
    eps = 0.000001
    smape = np.zeros(len(actual))
    for i, val in enumerate(zip(actual, fitted)):
        smape[i] = np.abs(val[0] - val[1]) / (eps + np.abs(val[0] + val[1]) / 2)
    return np.mean(smape)


@njit
def get_seasonality_weights(y, seasonal_period):
    seasonal_sample_weights = []
    weight = 0
    for i in range(len(y)):
        if (i) % seasonal_period == 0:
            weight += 1
        seasonal_sample_weights.append(weight)
    return seasonal_sample_weights


def cross_validation(y, test_size, n_splits, model_obj, step_size=1, **kwargs):
    mses = []
    maes = []
    mapes = []
    smapes = []
    residuals = []
    param_keys = list(kwargs.keys())
    if 'X' in param_keys:
        exogenous = kwargs['X'].copy()
        kwargs.pop('X')
    else:
        exogenous = None
    for split in range(n_splits):
        train_y = y[:-(split*step_size + test_size)]
        test_y = y[len(train_y): len(train_y) + test_size]
        if exogenous is not None:
            train_X = exogenous[:-(split*step_size + test_size), :]
            test_X = exogenous[len(train_y): len(train_y) + test_size, :]
        else:
            train_X = None
            test_X = None
        model_obj.fit(train_y, X=train_X, **kwargs)
        prediction = model_obj.predict(test_size, X=test_X)
        mses.append(calc_mse(test_y, prediction))
        maes.append(calc_mae(test_y, prediction))
        mapes.append(calc_mape(test_y, prediction))
        smapes.append(calc_smape(test_y, prediction))
        residuals.append(test_y - prediction)
    return {'mse': mses,
            'mae': maes,
            'mape': mapes,
            'smape': smapes,
            'residuals': residuals}

def logic_check(keys_to_check, keys):
    return set(keys_to_check).issubset(keys)

def logic_layer(param_dict):
    keys = param_dict.keys()
    # if param_dict['n_changepoints'] is None:
    #     if param_dict['decay'] != -1:
    #         return False
    if logic_check(['seasonal_period', 'max_rounds'], keys):
        if param_dict['seasonal_period'] is None:
            if param_dict['max_rounds'] < 4:
                return False
    if logic_check(['smoother', 'ma'], keys):
        if param_dict['smoother']:
            if param_dict['ma'] is not None:
                return False
    if logic_check(['seasonal_period', 'seasonality_weights'], keys):
        if param_dict['seasonality_weights']:
            if param_dict['seasonal_period'] is None:
                return False
    return True

def default_configs(seasonal_period, configs=None):
    if configs is None:
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
            configs = {
                # 'decay': [.01, .99, -1],
                # 'n_changepoints': [None, .25],
                # 'moving_medians': [True, False],
                # # 'trend_penalty': [True, False],
                # 'multiplicative': [True, False],
                'smoother': [True, False],
                # 'cov_threshold': [.7, -1],
                # 'ma': [None],
                'max_rounds': [3, 20],
                'seasonal_period': [None, seasonal_period],
                # 'seasonal_lr': [.1, .4, .6, .8, .9, 1]
                # 'rs_lr': [1, .9, .5, .1, 0],
                # 'linear_lr': [1, .9, .5, .1, 0],
                # 'alpha': [0.01, .1, .2, .3, .5, 1]
                }
        else:
            configs = {
                # 'decay': [.01, .99, -1],
                # 'n_changepoints': [None, .1],
                # 'moving_medians': [True, False],
                # 'trend_penalty': [True, False],
                'smoother': [True, False],
                'cov_threshold': [.5, -1],
                # 'ma': [None],
                'max_rounds': [5, 20],
                'seasonal_period': [None],
                # 'ets_lr': [1, .9, .5, .1, 0],
                # 'linear_lr': [1, .9, .5, .1, 0],
                # 'alpha': [0.01, .1, .2, .3, .5, 1]
                }
    keys = configs.keys()
    combinations = itertools.product(*configs.values())
    ds = [dict(zip(keys,cc)) for cc in combinations]
    ds = [i for i in ds if logic_layer(i)]
    return ds


