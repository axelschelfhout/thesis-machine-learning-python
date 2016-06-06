import pandas as pd

from random import randint
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #
# Data preparation

big_file = 'fakenames50k.csv'
small_file = 'fakeprofiledataset.csv'

train_file = big_file
test_file = small_file

# The header is in the first row. By adding header=0 we tell pd.read_csv it is
train_df = pd.read_csv(train_file, header=0, encoding='utf-8-sig', engine='python')
test_df = pd.read_csv(test_file, header=0, encoding='utf-8-sig', engine='python')

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

print(train_df.describe())

# The data frames values function will make this in to a Numpy array so we can use it in sklearn.
train_data = train_df.values
test_data = test_df.values
evaluate_data = evaluate_df.values
print(type(train_data))
# From https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii


train, test = train_test_split(train_data, train_size=0.7)
svm = SVC()
svm.fit(train[0:1000, :4], train[0:1000, 4])

print(test[5,])
print(svm.predict([test[5, :4]]))


def score_svm_model_by_iterations(data, iterations, train_size):
    model_score = []
    for i in range(0, iterations):
        rand = randint(0, len(data))
        x_train, x_test = train_test_split(data, train_size=train_size, random_state=rand)

        f_svm = SVC()
        f_svm.fit(x_train[0:2500, :4], x_train[0:2500, 4])

        this_score = f_svm.score(x_test[0::, :4], x_test[0::, 4:5])
        model_score.append(this_score)
    return sum(model_score)/len(model_score)

score_model_accuracy = score_svm_model_by_iterations(train_data, 10, 0.66)
print(score_model_accuracy)


