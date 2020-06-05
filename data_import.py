#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# Standardization, cleaning, padding and embeddings - All Processing Functions

import random
import pandas
import re
import datetime
import os
import ujson as json
import gensim
from keras.preprocessing.text import text_to_word_sequence
import datefinder
from numpy import argmax
from pathlib import Path

f_news_raw = "./news.txt"
f_news_splited_time = "./news_splited.txt"
f_news_without_stop = "./news_without_stops.txt"
f_news_padded = "news_padded.txt"
f_news_vector = "w2v_data.txt"
f_google_vecs = './data/googlenews_word_vectors/GoogleNews-vectors-negative300.bin'
null_word = 'nilwrd'
pad_size = 200
embedding_size = 100
str_filter = '!"#$%&%()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n'
file_out = 'all_news_30min'


def main():
    for news_dir in ['../data/bloomberg', '../data/reuters']:
        load_news(news_dir, news_dir.split('/')[-1])


def static_vars(**kwargs):
    """ Decorator static vars class"""

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(counter=1)
def load_news(sPath, name, n=None):
    """ Load News from repositories"""

    fOut = open(file_out, "a", encoding="utf8")

    for sChild in os.listdir(sPath):
        if n is not None and load_news.counter > n:
            break
        sChildPath = os.path.join(sPath, sChild)
        if os.path.isdir(sChildPath):
            load_news(sChildPath, name, n)
        else:
            try:
                n2 = open(sChildPath, 'r').readline().replace("--", "").strip()
                text = Path(sChildPath).read_text().replace('Z', ' ')
                re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            except UnicodeDecodeError:
                print(load_news.counter, sChildPath, "unicode error")
                continue
            if len(n2) == 0:
                news = ' '.join(sChildPath.split('/')[-1].split('-'))
            else:
                news = n2

            matches = list(datefinder.find_dates(text, strict=True))

            if len(matches) > 0:
                for date in matches:
                    if date.hour or date.minute or date.second:
                        break
            else:
                continue

            # news = ' '.join(text_to_word_sequence(news, filters=str_filter))
            load_news.counter += 1
            fOut.write('{0}\t{1}\t{2}\n'.format(date, name[0], news))

    fOut.close()


def clean_stopwords(file_in, file_out):
    """Cleaning stopwords and numbers"""

    fCorpus = open(file_in, "r", encoding="utf8").readlines()
    fOut = open(file_out, "a", encoding="utf8")
    fStop = open("./data/stopwords.txt", "r").read().split()
    for line in fCorpus:
        line = line.split()
        day = line[0]
        line = ' '.join(line[1:])
        words = []
        for word in line.split():
            word = re.sub(pattern="(\d|\"|\'|”|“|’|‘)*", repl="", string=word.strip()).strip()
            if (len(word) != 0) and word not in fStop:
                words.append(word)
        fOut.write('{0} {1}\n'.format(day, " ".join(words)))
    fOut.close()


def pad_news(file_in, file_out):
    """"Pad news to pad_size given size"""

    with open(file_out, "a", encoding="utf8") as fPadded:
        with open(file_in, "r", encoding="utf8") as f:
            for line in f:
                line = line.split()
                date = str(line[0]).strip()
                seq = line[1:]
                random.shuffle(seq, random.random)
                seq = seq[:pad_size]

                if len(seq) < pad_size:
                    pad = [null_word] * (pad_size - len(seq))
                    seq += pad

                fPadded.write(date + ' ' + ' '.join(seq) + '\n')


def load_google_news(pre_trainned_vecs):
    """ Load Google's pre-trained Word2Vec model."""

    model = gensim.models.KeyedVectors. \
        load_word2vec_format(pre_trainned_vecs, binary=True)
    print('Loaded %s word vectors.' % len(model.vocab))
    return model


def news_embeddings(file_in, file_out, model):
    """Create the news (word sequence) embeddings"""

    with open(file_out, 'a', encoding="utf8") as fData:
        with open(file_in, 'r', encoding="utf8") as fNews:
            # head = [next(fNews) for x in range(500)]
            for line in fNews:
                line = line.split()
                word_vectors = []
                for word in line[1:]:
                    vec = []
                    try:
                        if word != null_word:
                            vec = model[word].tolist()
                        else:
                            raise KeyError
                    except KeyError:
                        vec = [0.0] * embedding_size
                    word_vectors.append(vec)

                jline = {'date': line[0], 'words': word_vectors}
                fData.write(json.dumps(jline) + '\n')


if __name__ == '__main__':
    main()

""""
* Pre-trained word and phrase vectors
*
* We are publishing pre-trained vectors trained on part of Google News dataset (about 100 billion words). The model
* contains 300-dimensional vectors for 3 million words and phrases. The phrases were obtained using a simple data-driven
* approach described in [2]. The archive is available here: GoogleNews-vectors-negative300.bin.gz.
*
* An example output of ./distance GoogleNews-vectors-negative300.bin:
*
* References
*
* [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector
*  Space. In Proceedings of Workshop at ICLR, 2013.
*
* [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and
*  Phrases and their Compositionality. In Proceedings of NIPS, 2013.
*
* [3] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word Representations.
*  In Proceedings of NAACL HLT, 2013.
"""
