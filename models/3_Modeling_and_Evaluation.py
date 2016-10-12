
# coding: utf-8

# In[9]:

import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv("../data/raw/train.csv.zip", compression="zip", usecols=['loss'])
train_final1 = pd.read_csv("../data/interim/train_binary_encoded.csv")
X=train_final1#.values
y=train['loss']#.values

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.6, test_size=0.4)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=3, scoring='mean_absolute_error')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('../models/tpot_pipeline.py')


# In[ ]:
