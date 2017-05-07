import operator
from functools import reduce

import pandas as pd


def clean_data():
    df = pd.read_table('sentences.txt', header=None)
    df = df.apply(lambda x: x.astype(str).str.lower()) \
        .apply(lambda x: x.astype(str).str.split('[^a-z]')) \
        .applymap(lambda x: list(filter(None, x)))
    return df


def words(df):
    df = df.apply(lambda x: reduce(operator.add, x)) \
        .apply(lambda x: list(set(x)))
    return dict(enumerate(df))


print(words(clean_data()))
