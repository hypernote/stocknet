#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# It will merge the price timeseries with the news info timeseries

import pandas as pd

dateparse_prices = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %H:%M')
dateparse_news = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
output_folder = "/data/asset_raw_splited_series"
# Stocks of Interest
stocks = ["AAPL", "AMZN", "CSCO", "F", "GOOGL", "IBM", "MSFT", "NFLX", "ORCL", "VZ"]

for stock in stocks:
    print("Loading news timeseries for {0}...".format(stock))
    news_timeserie = pd.read_csv('../data/asset_rated_news/{0}.csv'.format(stock), sep='\t', parse_dates=['datetime'],
                                 date_parser=dateparse_news,
                                 names=['datetime', 'src', 'pos', 'neg', 'lit', 'unc', 'con', 'words'])

    first_new_date = news_timeserie.datetime[0]
    last_new_date = news_timeserie.datetime.iloc[-1]

    print("Loading SEE timeseries for {0}...".format(stock))
    see_timeserie = pd.read_csv('../data/asset_oie_splited/{0}.csv'.format(stock), sep='\t', parse_dates=['datetime'],
                                date_parser=dateparse_news,
                                names=['datetime', 'see'])

    print("Loading price timeseries for {0}...".format(stock))
    price_timeserie = pd.read_csv('../data/intraday_top50/{0}.txt'.format(stock),
                                  parse_dates={'datetime': ['date', 'time']},
                                  date_parser=dateparse_prices,
                                  names=['date', 'time', 'open', 'max', 'min', 'close', 'vol'])

    price_timeserie = price_timeserie[first_new_date < price_timeserie.datetime]
    price_timeserie = price_timeserie[price_timeserie.datetime < last_new_date]

    print("Performing timeseries merge for {0}...".format(stock))
    # Merging News with SEE timeseries
    nlp_df = pd.merge(news_timeserie, see_timeserie, on='datetime', how="left", sort=True)

    # Merging News and SEE with price timeseries
    df = pd.merge(price_timeserie, nlp_df, on='datetime', how="left", sort=True)
    df.pos[df.pos.isnull()] = 0.0
    df.neg[df.neg.isnull()] = 0.0
    df.lit[df.lit.isnull()] = 0.0
    df.unc[df.unc.isnull()] = 0.0
    df.con[df.con.isnull()] = 0.0
    df.see[df.see.isnull()] = 0.0
    df['delta'] = (df.close - df.close.shift(1)).fillna(0)

    # Removing unused attributes
    df.drop('src', axis=1, inplace=True)
    df.drop('words', axis=1, inplace=True)

    # Removing other attibutes for a more fair comparison with baseline
    df.drop('open', axis=1, inplace=True)
    df.drop('max', axis=1, inplace=True)
    df.drop('min', axis=1, inplace=True)
    df.drop('vol', axis=1, inplace=True)

    print("Wrinting {0} merged series to output folder '{1}'".format(stock, output_folder))
    df.to_pickle("{0}/{1}.pkl".format(output_folder, stock))
