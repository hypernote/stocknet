#!/usr/bin/env python
# -*- coding: utf-8 -*-
# jandersonborges@gmail.com
# OIE Extraction and vector sum for use in SEE baseline method
# The reference SEE paper used ReVerb to extract OIE http://reverb.cs.washington.edu/
import subprocess
import gensim
import numpy as np

extract_rel = False
data_folder = "/home/janderson/dev/mestrado/data"
word_vector_path = "/data/GoogleNews-vectors-negative300.bin"

def extract_relations(s):
# Put your ReVerb jar in the same folder
    """ Extracts OpenIE relations using Reverb (As described in paper)"""
    r = subprocess.Popen("echo '{0}' | java -Xmx512m -jar reverb-latest.jar".format(s),
                         shell=True, stdout=subprocess.PIPE).stdout.read()
    oargs = r.decode("utf-8").strip().split('\t')
    return " ".join(oargs[-3:])


model = gensim.models.KeyedVectors. \
    load_word2vec_format(word_vector_path, binary=True)
print('Loaded %s word vectors.' % len(model.vocab))

stocks = ["AAPL", "AMZN", "CSCO", "F", "GOOGL", "IBM", "MSFT", "NFLX", "ORCL", "VZ"]

for stock in stocks:

    count = 0

    fOut = open("{0}/asset_oie_splited/{1}.csv".format(data_folder, stock), "a+")
    for line in open("{0}/asset_rated_news/{1}.csv".format(data_folder, stock), "r"):
        headline = line.strip().split("\t")[-1]

        try:
            if extract_rel:
                extracted = extract_relations(headline)
            else:
                extracted = headline
        except:
            continue

        try:

            M = []

            for w in extracted.split():
                try:
                    v = model[w]
                    M.append(v)
                except:
                    print("[{0}] not found".format(w))
            matrix_sum = np.sum(np.array(M))

            fOut.write("\t".join(line.strip().split("\t")[:1]) + "\t" + str(matrix_sum) + "\n")
        except:
            fOut.write("\t".join(line.strip().split("\t")[:1]) + "\t" + str(0.0) + "\n")

        count += 1

    fOut.close()
    print(str(count) + " headlines analysed for " + stock)
