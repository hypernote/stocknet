# StockNet's Default Implementation

This repository contains the default partial but functional python's implementation of StockNet.

WARNING: Clean code needed (Hard coded references and bad code pratices ahead)

## About StockNet

StockNet is a novel deep neural architecture to predict stock prices based on Temporal Convolutional Networks and inspired by acoustic models for voice synthesis. 

For more information about the full implementation and architecture, please check my [master thesis here](https://tede.ufam.edu.br/bitstream/tede/7409/6/Disserta%C3%A7%C3%A3o_JandersonNascimento_PPGI.pdf).

## Dataset

The News dataset is the same used in [Ding et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014148.pdf).

- [Ding et al., 2014] Xiao Ding, Yue Zhang, Ting Liu, and Junwen Duan. Using structured events to predict stock price movement: An empirical investigation. In Proc. of EMNLP, pages 1415–1425, Doha, Qatar, October 2014. Association for Computational Linguistics.

The fulll Stock Prices dataset is an extensive 30 minute window stock prices dataset including the 10 more liquid stocks extracted from Dow Jones and NASDAQ from January, 2004 to August 2018 and can be requested from the author if you need. You also can use any price timeseries you want in StockNet.

## Pre-requisites

Python Version: 3.7 

The python pre-requisites can be found at requirements.txt file. Just install them by:

```pip install -r requirements.txt```

## How to run

After configure the folderes, news and price datasets paths in scripts, just run them the following sequence:

- data_import.py
- clean_dates.py
- break_news_30min.py
- sentiment_extraction.py
- split_assets.py
- oie_extraction.py (if using see)
- merge_price_news_w2v.py

- run_experiments.py

## Citing StockNet

If you think StockNet is useful for your work, please consider citing us.
```BibTeX
@article{nascimento2019stocknet,
  title={StockNet: A Multivariate Deep Neural Architecture for stock prices prediction},
  author={NASCIMENTO, Janderson Borges},
  school = {Universidade Federal do Amazonas},
  address     =  {Manaus, Brazil},  
  year         = 2019,  
  month        = 9,
  publisher={Universidade Federal do Amazonas},
  url       = {https://tede.ufam.edu.br/handle/tede/7409}
  }
```


