#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# StockNet evaluation experiments in GPU

from __future__ import print_function, division
import numpy as np
np.random.seed(42)
from models import StockNet, WaveNet
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

# Solve TensorFlow Problem in GPU Memory Growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

timeseries_folder = "/data/asset_raw_splited_series"
stocks = ["AAPL", "AMZN", "CSCO", "F", "GOOGL", "IBM", "MSFT", "NFLX", "ORCL", "VZ"]


def toarray(ts):
    timeseries = np.atleast_2d(np.asarray(ts))
    if timeseries.shape[0] == 1:
        timeseries = timeseries.T
    return timeseries


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_prediction(timeseries, forecast_horizon=1, n_epochs=3000, stock_net_active=False, train_model=True,
                        stock_name=""):
    length = len(timeseries) - forecast_horizon

    # Input attributes
    if stock_net_active:
        series = toarray(timeseries[['close', 'pos', 'neg', 'lit', 'unc', 'con']].values)
    else:
        series = toarray(timeseries[['close']].values)

    nb_features = series.shape[1]

    # prime model
    X = series[:-forecast_horizon].reshape(1, length, nb_features)
    y = toarray(timeseries[['close']].values)[forecast_horizon:].reshape(1, length, 1)

    if stock_net_active:
        model = StockNet(length, nb_features)
    else:
        model = WaveNet(length)

    best_model_file = '../data/models/{0}_best_model_{1}.h5'.format("stocknet" if stock_net_active else "wavenet",
                                                                    stock_name)
    if train_model:
        checkpoint = ModelCheckpoint(best_model_file, verbose=0, monitor='mean_squared_error',
                                     save_best_only=True, mode='auto')
        model.fit(X, y, epochs=n_epochs, callbacks=[checkpoint], verbose=False)

    model.load_weights(best_model_file)

    # get fit for training data
    X_val = series[forecast_horizon:].reshape(1, length, nb_features)
    X_trfit = model.predict([X_val])

    y_hat = model.predict([X_val])[0, -forecast_horizon:, 0]

    return y_hat, X_trfit.flatten()


# Starting Evaluation
for stock in stocks:
    # how many timeframes will be predicted
    predict_horizon = 10
    # how many train epochs
    epochs = 1500
    # how many timeframes will be showed in graph
    plot_horizon = 100

    timeserie = pd.read_pickle('{0}/{1}.pkl'.format(timeseries_folder,stock))

    train = timeserie.iloc[:-predict_horizon]
    test = timeserie.iloc[-predict_horizon:]

    # Interpolatind last X with first y
    y_series = test.append(train.iloc[-1]).close
    y_series.sort_index(inplace=True)
    plt.plot(train.close[-plot_horizon:], 'k-', label='past', linewidth=.5)
    plt.plot(y_series, ':', label='real', color='0.4', linewidth=.6)

    # Running WaveNet ******
    y_hat_wn, X_fit = evaluate_prediction(train, predict_horizon, n_epochs=epochs, stock_name=stock)
    # rmse_wn_str = " {:.4f}".format(np.sqrt(mean_squared_error(y_hat_wn, test.close)))
    mape_wn_str = " {:.4f}".format(mape(y_hat_wn, test.close))
    print("Wavenet results for {0} RMSE: {1} MAPE: {2}".format(stock, rmse_wn_str, mape_wn_str))
 
    # Converting y_hat to Dataframe and Interpolatind last X with first y_hat
    y_hat_wn_plot = pd.Series(y_hat_wn, index=test.index)
    y_hat_wn_plot = y_hat_wn_plot.append(pd.Series([train.iloc[-1].close], index=[train.index[-1]]))
    y_hat_wn_plot.sort_index(inplace=True)

    # Running StockNet ******
    y_hat_sn, _ = evaluate_prediction(train, predict_horizon, n_epochs=epochs, stock_net_active=True, train_model=True,
                                      stock_name=stock)
    sn_rmse_str = " {:.4f}".format(np.sqrt(mean_squared_error(y_hat_sn, test.close)))
    mape_sn_str = " {:.4f}".format(mape(y_hat_sn, test.close))

    # t, p = stats.ttest_ind(y_hat_sn, y_hat_wn)
    # print("StockNet/WaveNet statistical tests: t-value={0} and p-value={1}".format(t, p))

    # Converting y_hat_sn to Dataframe and Interpolatind last X with first y_hat_sn
    y_hat_sn_plot = pd.Series(y_hat_sn, index=test.index)
    y_hat_sn_plot = y_hat_sn_plot.append(pd.Series([train.iloc[-1].close], index=[train.index[-1]]))
    y_hat_sn_plot.sort_index(inplace=True)

    # Running SSE (Baseline) ******
    # 7 lag variables (according paper)
    see_timeserie = series_to_supervised(timeserie[['close', 'see']], n_in=7)

    # Splitting Train/test
    X_train = see_timeserie.iloc[:-predict_horizon]
    X_test = see_timeserie.iloc[-predict_horizon:]
    y_train = X_train['var1(t)']
    y_test = X_test['var1(t)']
    X_train.drop(['var1(t)'], axis=1, inplace=True)
    X_test.drop(['var1(t)'], axis=1, inplace=True)

    # 1000 estimators with max 50 nodes (according SEE paper)
    model = RandomForestRegressor(n_estimators=1000, max_leaf_nodes=50, n_jobs=8, random_state=42)
    model.fit(X_train, y_train)
    y_hat_see = model.predict(X_test)
    see_rmse_str = " {:.4f}".format(np.sqrt(mean_squared_error(y_hat_see, test.close)))
    mape_see_str = " {:.4f}".format(mape(y_hat_see, test.close))
    print("SEE-RMSE SEE-MAPE SN-RMSE SN-MAPE (Results for {0})".format(stock))
    print("{0} {1} {2} {3} {4}".format("[Win]" if float(see_rmse_str) >= float(sn_rmse_str) else "[Lost]",
                                       see_rmse_str, mape_see_str, sn_rmse_str, mape_sn_str))

    # Statistical Tests (null hypothesis ->  y_hat_sn eq v_random)
    random_scale = (np.max(train.iloc[-predict_horizon:].close) - np.min(train.iloc[-predict_horizon:].close)) / 2
    v_random = np.random.normal(loc=train.iloc[-1].close, scale=random_scale, size=predict_horizon)
    t, p = stats.ttest_ind(y_hat_sn, v_random)
    print("Statistical tests: t-value={0} and p-value={1} SEE Mean={2} SN Mean={3}".format(t, p, np.mean(y_hat_sn),
                                                                                           np.mean(v_random)))
    print("Shapiro SN {0}".format(stats.shapiro(y_hat_sn)))
    print("Shapiro Random {0}".format(stats.shapiro(v_random)))

    # Converting y_hat_see to Dataframe and Interpolatind last X with first y_hat_see
    y_hat_see_plot = pd.Series(y_hat_see, index=test.index)
    y_hat_see_plot = y_hat_see_plot.append(pd.Series([train.iloc[-1].close], index=[train.index[-1]]))
    y_hat_see_plot.sort_index(inplace=True)

    # Setting up Plotting
    plt.plot(y_hat_see_plot, 'r:', label='SEE Baseline (RMSE:{0})'.format(see_rmse_str), linewidth=.6)
    # plt.plot(y_hat_wn_plot, 'b:', label='WaveNet (RMSE:{0})'.format(rmse_wn_str), linewidth=.5)
    plt.plot(y_hat_sn_plot, 'g:', label='StockNet (RMSE:{0})'.format(sn_rmse_str), linewidth=.6)

    plt.legend(loc='upper left')
    plt.title("Last " + str(plot_horizon) + " samples and " + str(predict_horizon) + "-step prediction for " + stock)
    # plt.savefig('../data/results/stocknet_{0}.svg'.format(stock))
    # plt.close()
