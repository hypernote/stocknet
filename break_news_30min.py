#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# Put news timeseries in 30min timeframe in order to be aligned to prices timeseries

from collections import OrderedDict

hNews = {}
max_size = 0

with open("../data/all_news_headlines_30min.csv", "a") as fOut:
    for line in open("../data/all_news_intraday_1", "r"):
        line = line.replace("-05:00", "")

        hour = int(line[11:13])
        minute = int(line[14:16])

        if minute > 30:
            shour = str(hour + 1).zfill(2)
            if shour == "24":
                shour = "00"
            sminute = "00:00"
        else:
            shour = line[11:13]
            sminute = "30:00"

        line = line[:11] + shour + line[13:]
        line = line[:14] + sminute + line[19:]

        # using this same loop to remove empty news
        fields = line.split('\t')
        if len(fields[-1]) == 0:
            continue
        w = fields[-1].strip()

        # Concatenating news from the same timestep
        if fields[0] in hNews:
            # if (len(hNews[fields[0]]['words']) + len(w)) < 151:
            hNews[fields[0]]['words'] += w
        else:
            hNews[fields[0]] = dict(type=fields[1], words=w)

    d = OrderedDict(sorted(hNews.items()))

    for key in d:
        if len(d[key]['words']) > max_size:
            max_size = len(d[key]['words'])
        s = "\t".join([key, d[key]['type'], d[key]['words']])
        fOut.write(s + '\n')

    print(max_size)
