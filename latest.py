import pandas as pd
import numpy as np

# The header is in the first row. By adding header=0 we tell pd.read_csv it is
df = pd.read_csv('fakeprofiledataset.csv', header=0)

# Transform the gender STRING to integer. So we can work with it. Female: 0, Male: 1
df.Gender = df.Gender.map({'female': 0, 'male': 1}).astype(int)

# Remove all the other Object types in the data frame.
dfdrop = df.dtypes[df.dtypes.map(lambda x: x == 'object')].keys()
df = df.drop(dfdrop, axis=1)
# Also remove this, we dont need it for now.
df = df.drop(['NationalID','Longitude','Latitude','WesternUnionMTCN','MoneyGramMTCN','TelephoneCountryCode','CCNumber','CVV2','Number'], axis=1)

# --
train_data = df.values
print(train_data)

