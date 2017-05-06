import pandas as pd


def clean_data():
    df = pd.read_table('sentences.txt', header=None)
    df = df.apply(lambda x: x.astype(str).str.lower()) \
        .apply(lambda x: x.astype(str).str.split('[^a-z]')) \
        .applymap(lambda x: list(filter(None, x)))

    print(df)


clean_data()
