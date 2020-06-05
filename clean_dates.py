#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# Extracts news published only in interest interval

fOut  = open('../data/all_news_intraday_1', 'a')

with open('../data/all_news_intraday', 'r') as f:
	prior = '0000'
	for line in f:		
		actual = line[:4]
		if int(actual.strip()) in range(2006, 2014):
			year = actual						
		else:
			year = prior
		prior = actual

		fOut.write(year+line[4:])

fOut.close()
