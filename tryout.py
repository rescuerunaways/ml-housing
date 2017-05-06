import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing


# from jedi.refactoring import inline


def tryOut():
    color = sns.color_palette()

    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', 500)

    train_df = pd.read_csv('train.csv')
    head = train_df.head()

    # Let us start with target variable exploration - 'price_doc'. \
    #  First let us do a scatter plot to see if there are any outliers in the data.
    #
    # plt.figure(figsize=(8, 6))
    # plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
    # plt.xlabel('index', fontsize=12)
    # plt.ylabel('price', fontsize=12)
    # plt.show()
    #

    # Looks okay to me. Also since the metric is RMSLE,
    # I think it is okay to have it as such. However if needed, one can truncate the high values,
    # We can now bin the 'price_doc' and plot it.

    # plt.figure(figsize=(12, 8))
    # sns.distplot(train_df.price_doc.values, bins=50, kde=True)
    # plt.xlabel('price', fontsize=12)
    # plt.show()

    # Certainly a very long right tail. Since our metric  \
    # is Root Mean Square **Logarithmic** error, let us plot the log of price_doc variable.


    # plt.figure(figsize=(12,8))
    # sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
    # plt.xlabel('price', fontsize=12)
    # plt.show()

    # This looks much better than the previous one.
    # Now let us see how the median housing price change with time.

    # train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
    # grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()
    #
    # plt.figure(figsize=(12,8))
    # sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
    # plt.ylabel('Median Price', fontsize=12)
    # plt.xlabel('Year Month', fontsize=12)
    # plt.xticks(rotation='vertical')
    # plt.show()


    # There are some variations in the median price with respect to time.
    # Towards the end, there seems to be some linear increase in the price values.
    # Now let us dive into other variables and see.
    # Let us first start with getting the count of different data types. 


    train_df = pd.read_csv('train.csv', parse_dates=['timestamp'])
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['Count', 'Column Type']
    dtype_df.groupby('Column Type').aggregate('count').reset_index()

    # So majority of them are numerical variables with 15 factor variables and 1 date variable
    # Let us explore the number of missing values in each column.

    missing_df = train_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count'] > 0]
    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel('Count of missing values')
    ax.set_title('Number of missing values in each column')
    plt.show()

    # Seems variables are found to missing as groups.
    # Since there are 292 variables,
    # let us build a basic xgboost model and then explore only the important variables.
    for f in train_df.columns:
        if train_df[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))

    train_y = train_df.price_doc.values
    train_X = train_df.drop(['id', 'timestamp', 'price_doc'], axis=1)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

    # plot the important features #
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()

    # So the top 5 variables and their description from the data dictionary are:
    # 
    #  1. full_sq - total area in square meters, including loggias, balconies and other non-residential areas
    #  2. life_sq - living area in square meters, excluding loggias, balconies and other non-residential areas
    #  3. floor - for apartments, floor of the building
    #  4. max_floor - number of floors in the building
    #  5. build_year - year built
    # 
    # Now let us see how these important variables are distributed with respect to target variable.
    # 
    # **Total area in square meters:**


tryOut()
