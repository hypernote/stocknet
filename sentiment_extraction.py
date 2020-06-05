#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# Standardization, cleaning and sentiment analysis

import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing import PorterStemmer
from nltk import RegexpStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine

f_google_vecs = "/data/GoogleNews-vectors-negative300.bin"
f_comp_vecs = "/data/word.vec"
input_news = "../data/all_news_headlines_30min.csv"
output_file = "../data/rated_headlines.csv"
loughran_path = "../data/Loughran-McDonald"

model = KeyedVectors.load(f_comp_vecs, mmap='r')


def stemmed_vector(word):
    try:
        return model[word], word
    except:
        try:
            stemmer = PorterStemmer()
            word = stemmer.stem(word)
            return model[word], word
        except:
            try:
                stemmer = SnowballStemmer("english")
                word = stemmer.stem(word)
                return model[word], word
            except:
                try:
                    l = WordNetLemmatizer()
                    word = l.lemmatize(word)
                    return model[word], word
                except:
                    stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
                    word = stemmer.stem(word)
                    try:
                        return model[word], word
                    except:
                        return np.array(300 * [0]), ""


def main():
    pos_words = set([w.strip() for w in open(loughran_path+"/positive.txt", "r").readlines()])
    pos_vecs = []
    neg_words = set([w.strip() for w in open(loughran_path+"/negative.txt", "r").readlines()])
    neg_vecs = []
    lit_words = set([w.strip() for w in open(loughran_path+"/litigious.txt", "r").readlines()])
    lit_vecs = []
    unc_words = set([w.strip() for w in open(loughran_path+"/uncertainty.txt", "r").readlines()])
    unc_vecs = []
    con_words = set([w.strip() for w in open(loughran_path+"/constraining.txt", "r").readlines()])
    con_vecs = []

    # Evaluating centroid vectors
    for word in pos_words:
        v, w = stemmed_vector(word)
        pos_vecs.append(v)
    pos_vecs = np.array(pos_vecs)
    print("positive vectors loaded", pos_vecs.shape)
    pos_centroid = np.mean(pos_vecs, axis=0)

    for word in neg_words:
        v, w = stemmed_vector(word)
        neg_vecs.append(v)
    neg_vecs = np.array(neg_vecs)
    print("negative vectors loaded", neg_vecs.shape)
    neg_centroid = np.mean(neg_vecs, axis=0)

    for word in lit_words:
        v, w = stemmed_vector(word)
        lit_vecs.append(v)
    lit_vecs = np.array(lit_vecs)
    print("litigious vectors loaded", lit_vecs.shape)
    lit_centroid = np.mean(lit_vecs, axis=0)

    for word in unc_words:
        v, w = stemmed_vector(word)
        unc_vecs.append(v)
    unc_vecs = np.array(unc_vecs)
    print("uncertainty vectors loaded", unc_vecs.shape)
    unc_centroid = np.mean(unc_vecs, axis=0)

    for word in con_words:
        v, w = stemmed_vector(word)
        con_vecs.append(v)
    con_vecs = np.array(con_vecs)
    print("constraining vectors loaded", con_vecs.shape)
    con_centroid = np.mean(con_vecs, axis=0)

    count = 0

    for line in open(input_news, "r"):
        published = line.strip().split("\t")
        head_line = published[-1]

        # split into words
        tokens = word_tokenize(head_line)
        words = [word.lower() for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        word_vectors = []
        for w in words:
            word_vectors.append(stemmed_vector(w)[0])
        if len(word_vectors) == 0:
            continue
        word_vectors = np.array(word_vectors)
        vMean = np.mean(word_vectors, axis=0)

        pos_rating = cosine(vMean, pos_centroid)
        neg_rating = cosine(vMean, neg_centroid)
        lit_rating = cosine(vMean, lit_centroid)
        unc_rating = cosine(vMean, unc_centroid)
        con_rating = cosine(vMean, con_centroid)

        with open(output_file, "a") as fOut:
            fOut.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(published[0], published[1], str(pos_rating),
                                                                         str(neg_rating), str(lit_rating),
                                                                         str(unc_rating), str(con_rating),
                                                                         " ".join(words)))

        count += 1

    print(str(count) + " headlines analysed")


if __name__ == '__main__':
    main()

