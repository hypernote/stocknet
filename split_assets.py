#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# Splits the news news by company using the ticket code

input_file = "/home/janderson/dev/mestrado/data/rated_headlines.csv"
output_dir = "/home/janderson/dev/mestrado/data/"
# This map news with interest tokens to the ticket
stocks = {"AAPL": ["aapl", "apple"],
          "AMZN": ["amzn", "amazon.com", "amazon"],
          "CSCO": ["csco", "cisco"],
          "F": ["f", "ford"],
          "GOOGL": ["goog", "googl", "alphabet", "google"],
          "IBM": ["ibm"],
          "MSFT": ["msft", "ms", "microsoft"],
          "NFLX": ["nflx", "netflix"],
          "ORCL": ["orcl", "oracle"],
          "INTC": ["intc", "intel"],
          "JPM": ["jpm", "jpmorgan"],
          "XOM": ["xom", "exxonmobil", "exxon", "oil", "mobil"],
          "QCOM": ["qcom", "qualcomm", "qualcom"],
          "WMT": ["wmt", "qualcomm", "qualcom"],
          "VZ": ["vz", "verizon"]}
# TODO: Should we add other related info? (eg. 'gmail' into GOOGL or 'processor' and 'amd' into INTC ???

count = 0

for line in open(input_file, "r"):
    headline = line.strip().split("\t")[-1]

    for s in stocks:

        if any(e in stocks[s] for e in headline.split()):
            file_name_out = output_dir + "asset_rated_news/{0}.csv".format(s)
            with open(file_name_out, "a") as fOut:
                fOut.write(line)

    count += 1

print(str(count) + " headlines splited in assets")
