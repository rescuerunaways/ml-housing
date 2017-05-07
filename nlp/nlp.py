import operator
from functools import reduce

import numpy as np
import pandas as pd
from scipy import spatial


def clean():
    df = pd.read_table('sentences.txt', header=None)
    df = df.apply(lambda x: x.astype(str).str.lower()) \
        .apply(lambda x: x.astype(str).str.split('[^a-z]')) \
        .applymap(lambda x: list(filter(None, x)))
    return df


def words(df):
    df = df.apply(lambda x: reduce(operator.add, x)) \
        .apply(lambda x: list(set(x)))
    return dict(enumerate(sorted(list(map(lambda x: x, df.to_dict()[0])))))


def matrix(sentns, wrds):
    sntncs = sentns[0].tolist()
    wrds = list(wrds.values())
    pr = list(map(lambda sentence: list(map(lambda word: sentence.count(word), wrds)), sntncs))
    m = np.matrix(pr)
    return m


def similar(m):
    result = list(map(lambda v: spatial.distance.cosine(m[0], v), m))
    result = dict(enumerate(result))
    sorted_x = sorted(result.items(), key=operator.itemgetter(1))
    return sorted_x


print(similar(matrix(clean(), words(clean()))))
