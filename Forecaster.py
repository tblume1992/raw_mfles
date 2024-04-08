# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm
import pandas as pd
import json
from MFLES.Conformal import get_interval
from MFLES.Model import median, ses_ensemble, ols, wls, fast_ols, lasso_nb, siegel_repeated_medians, OLS
from MFLES.FeatureEngineering import (get_basis,get_future_basis,get_fourier_series)
from MFLES.utils import (calc_cov,calc_seas_strength,calc_trend_strength,
                       cap_outliers,calc_rsq,calc_mse, cross_validation,
                       set_fourier, default_configs, get_seasonality_weights)
tqdm.pandas()



def fit_mfles(df,
              id_column,
              time_column,
              value_column,
              forecast_horizon,
              freq,
              verbose,
              **kwargs
              ):
    y = df[value_column].values
    if 'seasonal_period' in kwargs.keys():
        if kwargs['seasonal_period'] is not None:
            season = [i for i in kwargs['seasonal_period'] if i * 1.5 < len(y)]
            if not season:
                season = None
            kwargs['seasonal_period'] = season
    mfles = MFLES(verbose=verbose)
    fitted = mfles.fit(y, **kwargs)
    predicted = mfles.predict(forecast_horizon)
    forecast_dates = pd.date_range(start=df[time_column].max() + pd.DateOffset(1),
                                  periods=forecast_horizon,
                                  freq=freq)
    forecast_df = pd.DataFrame({id_column: df[id_column].iloc[0],
                                time_column: forecast_dates,
                                'mfles': predicted})
    df['mfles'] = fitted
    return pd.concat([df, forecast_df])

def fit_from_df(df,
                forecast_horizon,
                freq,
                floor=None,
                metric='mse',
                id_column='unique_id',
                time_column='ds',
                value_column='y',
                verbose=1,
                **kwargs):
    df = df.copy()
    df = df[[id_column, time_column, value_column]]
    if 'seasonal_period' in kwargs.keys():
        if kwargs['seasonal_period'] is not None:
            if not isinstance(kwargs['seasonal_period'], list):
                kwargs['seasonal_period'] = [kwargs['seasonal_period']]
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    output = df.groupby(id_column).progress_apply(fit_mfles,
                                                  id_column=id_column,
                                                  time_column=time_column,
                                                  value_column=value_column,
                                                  forecast_horizon=forecast_horizon,
                                                  freq=freq,
                                                  verbose=verbose,
                                                  **kwargs)
    if floor is not None:
        output['mfles'] = output['mfles'].clip(lower=floor)
    return output

def opt_mfles(df,
              id_column,
              time_column,
              value_column,
              freq,
              seasonal_period,
              forecast_horizon,
              test_size,
              n_steps,
              metric,
              params,
              verbose,
              step_size
              ):
    y = df[value_column].values
    if seasonal_period is not None:
        season = [i for i in seasonal_period if i * 1.5 < len(y)]
        if not season:
            season = None
    else:
        season = None
    mfles = MFLES(verbose=verbose)
    opt_param = mfles.optimize(y,
                               seasonal_period=season,
                               test_size=test_size,
                               n_steps=n_steps,
                               metric=metric,
                               params=params,
                               step_size=step_size)
    fitted = mfles.fit(y, **opt_param)
    predicted = mfles.predict(forecast_horizon)
    df['mfles'] = fitted
    forecast_dates = pd.date_range(start=df[time_column].max() + pd.DateOffset(1),
                                  periods=forecast_horizon,
                                  freq=freq)
    forecast_df = pd.DataFrame({id_column: df[id_column].iloc[0],
                                time_column: forecast_dates,
                                'mfles': predicted})
    df['mfles'] = fitted
    df['opt_param'] = json.dumps(opt_param)
    return pd.concat([df, forecast_df])

def optimize_from_df(df,
                     forecast_horizon, 
                     test_size, 
                     n_steps,
                     freq,
                     step_size=1,
                     params=None,
                     seasonal_period=None,
                     floor=None,
                     metric='mse',
                     id_column='unique_id',
                     time_column='ds',
                     value_column='y',
                     verbose=1):
    if seasonal_period is not None:
        if not isinstance(seasonal_period, list):
            seasonal_period = [seasonal_period]
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    output =  df.groupby(id_column).progress_apply(opt_mfles,
                                                   id_column=id_column,
                                                   time_column=time_column,
                                                   value_column=value_column,
                                                   freq=freq,
                                                   seasonal_period=seasonal_period,
                                                   forecast_horizon=forecast_horizon,
                                                   test_size=test_size,
                                                   n_steps=n_steps,
                                                   metric=metric,
                                                   params=params,
                                                   verbose=verbose,
                                                   step_size=step_size)
    if floor is not None:
        output['mfles'] = output['mfles'].clip(lower=floor)
    return output


class MFLES:
    def __init__(self, verbose=1, robust=None):
        self.penalty = None
        self.trend = None
        self.seasonality = None
        self.robust = robust
        self.const = None
        self.aic = None
        self.upper = None
        self.lower= None
        self.exogenous_models = None
        self.verbose = verbose
        self.predicted = None

    def fit(self,
            y,
            seasonal_period=None,
            X=None,
            fourier_order=None,
            ma=None,
            alpha=.1,
            decay=-1,
            n_changepoints=.25,
            seasonal_lr=.9,
            rs_lr=1,
            exogenous_lr=1,
            exogenous_estimator=OLS,
            exogenous_params={},
            linear_lr=.9,
            cov_threshold=.7,
            moving_medians=False,
            max_rounds=50,
            min_alpha=.05,
            max_alpha=1.0,
            round_penalty=0.0001,
            trend_penalty=True,
            multiplicative=None,
            changepoints=True,
            smoother=False,
            seasonality_weights=False):
        """
        

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        seasonal_period : TYPE, optional
            DESCRIPTION. The default is None.
        fourier_order : TYPE, optional
            DESCRIPTION. The default is None.
        ma : TYPE, optional
            DESCRIPTION. The default is None.
        alpha : TYPE, optional
            DESCRIPTION. The default is .1.
        decay : TYPE, optional
            DESCRIPTION. The default is -1.
        n_changepoints : TYPE, optional
            DESCRIPTION. The default is .25.
        seasonal_lr : TYPE, optional
            DESCRIPTION. The default is .9.
        rs_lr : TYPE, optional
            DESCRIPTION. The default is 1.
        linear_lr : TYPE, optional
            DESCRIPTION. The default is .9.
        cov_threshold : TYPE, optional
            DESCRIPTION. The default is .7.
        moving_medians : TYPE, optional
            DESCRIPTION. The default is False.
        max_rounds : TYPE, optional
            DESCRIPTION. The default is 10.
        min_alpha : TYPE, optional
            DESCRIPTION. The default is .05.
        max_alpha : TYPE, optional
            DESCRIPTION. The default is 1.0.
        trend_penalty : TYPE, optional
            DESCRIPTION. The default is True.
        multiplicative : TYPE, optional
            DESCRIPTION. The default is None.
        changepoints : TYPE, optional
            DESCRIPTION. The default is True.
        smoother : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        if cov_threshold == -1:
            cov_threshold = 10000
        n = len(y)
        if n < 4 or np.all(y == np.mean(y)):
            if self.verbose:
                if n < 4:
                    print('series is too short (<4), defaulting to naive')
                else:
                    print(f'input is constant with value {y[0]}, defaulting to naive')
            self.trend = np.append(y[-1], y[-1])
            self.seasonality = np.zeros(len(y))
            self.trend_penalty = False
            self.mean = 0
            self.std = 0
            return np.tile(y[-1], len(y))
        og_y = y
        self.og_y = og_y
        y = y.copy()
        if n_changepoints is None:
            changepoints = False
        if isinstance(n_changepoints, float) and n_changepoints < 1:
            n_changepoints = int(n_changepoints * n)
        self.linear_component = np.zeros(n)
        self.seasonal_component = np.zeros(n)
        self.ses_component = np.zeros(n)
        self.median_component = np.zeros(n)
        self.exogenous_component = np.zeros(n)
        self.exogenous_lr = exogenous_lr
        self.exo_model = []
        self.round_cost = []
        if multiplicative is None:
            if seasonal_period is None:
                multiplicative = False
            else:
                multiplicative = True
            if min(y) <= 0:
                multiplicative = False
        if multiplicative:
            self.const = y.min()
            y = np.log(y)
        else:
            self.const = None
            self.std = np.std(y)
            self.mean = np.mean(y)
            y = (y - self.mean) / self.std
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
        self.trend_penalty = trend_penalty
        if moving_medians and seasonal_period is not None:
            fitted = median(y, max(seasonal_period))
        else:
            fitted = median(y, None)
        self.median_component += fitted
        self.trend = np.append(fitted.copy()[-1:], fitted.copy()[-1:])
        mse = None
        equal = 0
        if ma is None:
            ma_cycle = cycle([1])
        else:
            if not isinstance(ma, list):
                ma = [ma]
            ma_cycle = cycle(ma)
        if seasonal_period is not None:
            seasons_cycle = cycle(list(range(len(seasonal_period))))
            self.seasonality = np.zeros(max(seasonal_period))
            fourier_series = []
            for period in seasonal_period:
                if fourier_order is None:
                    fourier = set_fourier(period)
                else:
                    fourier = fourier_order
                fourier_series.append(get_fourier_series(n,
                                                    period,
                                                    fourier))
            if seasonality_weights:
                cycle_weights = []
                for period in seasonal_period:
                    cycle_weights.append(get_seasonality_weights(y, period))
        else:
            self.seasonality = None
        for i in range(max_rounds):
            resids = y - fitted
            if mse is None:
                mse = calc_mse(y, fitted)
            else:
                if mse <= calc_mse(y, fitted):
                    if equal == 6:
                        break
                    equal += 1
                else:
                    mse = calc_mse(y, fitted)
                self.round_cost.append(mse)
            if seasonal_period is not None:
                seasonal_period_cycle = next(seasons_cycle)
                if seasonality_weights:
                    seas = wls(fourier_series[seasonal_period_cycle],
                               resids,
                               cycle_weights[seasonal_period_cycle])
                else:
                    seas = ols(fourier_series[seasonal_period_cycle],
                               resids)
                seas = seas * seasonal_lr
                component_mse = calc_mse(y, fitted + seas)
                if mse > component_mse:
                    mse = component_mse
                    fitted += seas
                    resids = y - fitted
                    self.seasonality += np.resize(seas[-seasonal_period[seasonal_period_cycle]:],
                                                  len(self.seasonality))
                    self.seasonal_component += seas
            if X is not None and i > 0:
                model_obj = exogenous_estimator(**exogenous_params)
                model_obj.fit(X, resids)
                self.exo_model.append(model_obj)
                _fitted_values = model_obj.predict(X) * exogenous_lr
                self.exogenous_component += _fitted_values
                fitted += _fitted_values
                resids = y - fitted
            if i % 2: #if even get linear piece, allows for multiple seasonality fitting a bit more
                if self.robust:
                    tren = siegel_repeated_medians(x=np.arange(n),
                                    y=resids)
                else:
                    if i==1 or not changepoints:
                        tren = fast_ols(x=np.arange(n),
                                        y=resids)
                    else:
                        cps = min(n_changepoints, int(.1*n))
                        lbf = get_basis(y=resids,
                                        n_changepoints=cps,
                                        decay=decay)
                        tren = np.dot(lbf, lasso_nb(lbf, resids, alpha=alpha))
                        tren = tren * linear_lr
                component_mse = calc_mse(y, fitted + tren)
                if mse > component_mse:
                    mse = component_mse
                    fitted += tren
                    self.linear_component += tren
                    self.trend += tren[-2:]
                    if i == 1:
                        self.penalty = calc_rsq(resids, tren)
            elif i > 4 and not i % 2:
                if smoother is None:
                    if seasonal_period is not None:
                        len_check = int(max(seasonal_period))
                    else:
                        len_check = 12
                    if resids[-1] > np.mean(resids[-len_check:-1]) + 3 * np.std(resids[-len_check:-1]):
                        smoother = 0
                    if resids[-1] < np.mean(resids[-len_check:-1]) - 3 * np.std(resids[-len_check:-1]):
                        smoother = 0
                    if resids[-2] > np.mean(resids[-len_check:-2]) + 3 * np.std(resids[-len_check:-2]):
                        smoother = 0
                    if resids[-2] < np.mean(resids[-len_check:-2]) - 3 * np.std(resids[-len_check:-2]):
                        smoother = 0
                    if smoother is None:
                        smoother = 1
                    else:
                        resids[-2:] = cap_outliers(resids, 3)[-2:]
                tren = ses_ensemble(resids,
                                    min_alpha=min_alpha,
                                    max_alpha=max_alpha,
                                    smooth=smoother*1,
                                    order=next(ma_cycle)
                                    )
                tren = tren * rs_lr
                component_mse = calc_mse(y, fitted + tren)
                if mse > component_mse + round_penalty * mse:
                    mse = component_mse
                    fitted += tren
                    self.ses_component += tren
                    self.trend += tren[-1]
            if i == 0: #get deasonalized cov for some heuristic logic
                if self.robust is None:
                    try:
                        if calc_cov(resids, multiplicative) > cov_threshold:
                            self.robust = True
                        else:
                            self.robust = False
                    except:
                        self.robust = True

            if i == 1:
                resids = cap_outliers(resids, 5) #cap extreme outliers after initial rounds
        if multiplicative:
            fitted = np.exp(fitted)
        else:
            fitted = self.mean + (fitted * self.std)
        self.multiplicative = multiplicative
        self.fitted = fitted
        return fitted

    def predict(self, forecast_horizon, X=None):
        last_point = self.trend[1]
        slope = last_point - self.trend[0]
        if self.trend_penalty and self.penalty is not None:
            slope = slope * max(0, self.penalty)
        self.predicted_trend = slope * np.arange(1, forecast_horizon + 1) + last_point
        if self.seasonality is not None:
            predicted = self.predicted_trend + np.resize(self.seasonality, forecast_horizon)
        else:
            predicted = self.predicted_trend
        if X is not None:
            for model in self.exo_model:
                predicted += model.predict(X) * self.exogenous_lr
        if self.const is not None:
            predicted = np.exp(predicted)
        else:
            predicted = self.mean + (predicted * self.std)
        self.predicted = predicted
        return predicted

    def optimize(self, y, test_size, n_steps, step_size=1, seasonal_period=None, metric='smape', params=None):
        """
        Optimization method for MFLES

        Parameters
        ----------
        y : np.array
            Your time series as a numpy array.
        test_size : int
            length of the test set to hold out to calculate test error.
        n_steps : int
            number of train and test sets to create.
        step_size : 1, optional
            how many periods to move after each step. The default is 1.
        seasonal_period : int or list, optional
            the seasonal period to calculate for. The default is None.
        metric : TYPE, optional
            supported metrics are smape, mape, mse, mae. The default is 'smape'.
        params : dict, optional
            A user provided dictionary of params to try. The default is None.

        Returns
        -------
        opt_param : TYPE
            DESCRIPTION.

        """
        configs = default_configs(seasonal_period, params)
        if len(y) - 5 < n_steps + test_size:
            n_steps = 1
            if len(y) - 5 < n_steps + test_size:
                test_size = int(.5 * test_size)
                if len(y) - 5 < n_steps + test_size:
                    test_size = int(.5 * test_size)
            if self.verbose:
                print(f'Series length too small, setting test_size to {test_size} and n_steps to {n_steps}')

        metrics = []
        for param in configs:
            try:
                cv_results = cross_validation(y,
                                              test_size,
                                              n_steps,
                                              MFLES(verbose=self.verbose),
                                              step_size=step_size,
                                              **param)
                metrics.append(np.mean(cv_results[metric]))
            except:
                metrics.append(10**10)
        opt_param = configs[np.argmin(metrics)]
        return opt_param

    def conformal(self, y, forecast_horizon, n_windows, coverage, future_X=None, **kwargs):
        intervals = get_interval(y,
                                 forecast_horizon,
                                 n_windows, coverage,
                                 self,
                                 **kwargs)
        self.fitted = self.fit(y, **kwargs)
        if future_X is None:
            self.predicted = self.predict(forecast_horizon)
        else:
            self.predicted = self.predict(forecast_horizon, X=future_X)
        if isinstance(coverage, list):
            upper = []
            lower = []
            for i in range(len(coverage)):
                upper.append(np.append(self.fitted + intervals[0][i],
                                       self.predicted + intervals[1][i]))
                lower.append(np.append(self.fitted - intervals[0][i],
                                       self.predicted - intervals[1][i]))
        else:
            upper = np.append(self.fitted + intervals[0],
                              self.predicted + intervals[1])
            lower = np.append(self.fitted - intervals[0],
                              self.predicted - intervals[1])
        self.upper = upper
        self.lower = lower
        return np.append(self.fitted, self.predicted), upper, lower

    def seasonal_decompose(self, y, **kwargs):
        fitted = self.fit(y, **kwargs)
        trend = self.linear_component
        exogenous = self.median_component + self.exogenous_component
        level = self.median_component + self.ses_component
        seasonality = self.seasonal_component
        if self.multiplicative:
            trend = np.exp(trend)
            level = np.exp(level)
            exogenous = np.exp(exogenous) - np.exp(self.median_component)
            if kwargs['seasonal_period'] is not None:
                seasonality = np.exp(seasonality)
            trend = trend * level
        else:
            trend = self.mean + (trend * self.std)
            level = self.mean + (level * self.std)
            exogenous = self.mean + (exogenous * self.std)
            if kwargs['seasonal_period'] is not None:
                seasonality = (seasonality * self.std)
            trend = trend + level - self.mean
        residuals = y - fitted
        self.decomposition = {'y': y,
                              'trend': trend,
                              'seasonality': seasonality,
                              'exogenous': exogenous,
                              'residuals': residuals
            }
        return self.decomposition

    def plot(self):
        plt.style.use("ggplot")
        if self.predicted is None:
            plt.plot(self.fitted, color='red', linestyle='dashed')
        else:
            plt.plot(np.append(self.fitted, self.predicted), color='red', linestyle='dashed')
        plt.plot(self.og_y, color='black')
        colors = ['royalblue', 'lightblue', 'lightsteelblue', 'lightgrey']
        if self.upper is not None:
            if not isinstance(self.upper, list):
                self.upper = [self.upper]
                self.lower = [self.lower]
            for i in range(len(self.upper)):
                colors = colors[:len(self.upper)]
                plt.fill_between(range(len(self.upper[0])),
                                         self.upper[i],
                                         self.lower[i],
                                         color=colors[i],
                                         alpha=.7)
        plt.show()

    def plot_decomposition(self):
        plt.style.use("ggplot")
        plots = 3
        if not np.all(self.exogenous_component == 0):
            plots += 1
        fig, ax = plt.subplots(plots)
        ax[0].plot(self.decomposition['trend'])
        ax[0].plot(self.decomposition['y'],
                   alpha=.5,
                   linestyle='dashed')
        ax[1].plot(self.decomposition['seasonality'])
        if self.multiplicative:
            detrended = self.decomposition['y'] / self.decomposition['trend']
        else:
            detrended = self.decomposition['y'] - self.decomposition['trend']
        ax[1].plot(detrended,
                   alpha=.5,
                   linestyle='dashed')
        cmap = plt.cm.Reds
        ax[-1].scatter(range(len(detrended)), y=self.decomposition['residuals'],
                   color=cmap(np.abs(self.decomposition['residuals']) / np.mean(np.abs(self.decomposition['residuals']))))
        ax[0].set_title('Trend')
        ax[1].set_title('Seasonality')
        if not np.all(self.exogenous_component == 0):
            ax[2].plot(self.decomposition['exogenous'])
            ax[2].set_title('Exogenous')
        ax[-1].set_title('Residuals')
        plt.tight_layout()
        plt.show()


