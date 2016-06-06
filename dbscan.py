import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #
# Data preparation

big_file = 'fakenames50k.csv'
small_file = 'fakeprofiledataset.csv'

train_file = big_file
test_file = small_file

# The header is in the first row. By adding header=0 we tell pd.read_csv it is
train_df = pd.read_csv(train_file, header=0, encoding='utf-8-sig', engine='python')
test_df = pd.read_csv(test_file, header=0, encoding='utf-8-sig', engine='python')

# Used df.dtypes to see what the column names are.
# print(train_df.dtypes)
# df['GenderTarget'] = 4

# Move gender from the first position, to the last position
cols = train_df.columns.tolist()
cols = cols[2:] + cols[:2]
train_df = train_df[cols]
test_df = test_df[cols]

# Transform the gender STRING to integer. So we can work with it. Female: 0, Male: 1
train_df.Gender = train_df.Gender.map({'female': 0, 'male': 1}).astype(int)
test_df.Gender = test_df.Gender.map({'female': 0, 'male': 1}).astype(int)

# Remove all the other Object types in the data frame.
df_drop = train_df.dtypes[train_df.dtypes.map(lambda x: x == 'object')].keys()
train_df = train_df.drop(df_drop, axis=1)
dropable_columns = ['NationalID', 'Longitude', 'Latitude', 'WesternUnionMTCN', 'MoneyGramMTCN', 'TelephoneCountryCode', 'CCNumber', 'CVV2', 'Number']
train_df = train_df.drop(dropable_columns, axis=1)

test_df = test_df.drop(df_drop, axis=1)
test_df = test_df.drop(dropable_columns, axis=1)

evaluate_df = test_df
# evaluate_df.Gender = evaluate_df.Gender.map({'female': 0, 'male': 1}).astype(int)
# test_df = test_df.drop(['Gender'], axis=1)

print(train_df.describe())

# The data frames values function will make this in to a Numpy array so we can use it in sklearn.
train_data = train_df.values
test_data = test_df.values
evaluate_data = evaluate_df.values
print(type(train_data))
# From https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii


np.random.shuffle(train_data)
db_scan = DBSCAN(eps=2, min_samples=3)
db_scan.fit(train_data)
# The amount of clusters found
print(len(np.unique(db_scan.labels_)))

