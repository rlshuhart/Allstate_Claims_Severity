
# coding: utf-8

# # Import Data

# In[2]:

import pandas as pd
from sklearn.model_selection import train_test_split
#train = pd.read_csv("../data/raw/train.csv.zip", compression="zip", usecols=['loss'])
train_binary = pd.read_pickle("../data/processed/train_binary_encoded.p")

X=train_binary.drop('loss', axis=1)
y=train_binary['loss']

# Chosing a lesser training size of 60% to avoid over fitting.
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.6, test_size=0.4)


# # Tree-based Pipeline Optimization Tool  (TPOT)
# Randal S. Olson, Ryan J. Urbanowicz, Peter C. Andrews, Nicole A. Lavender, La Creis Kidd, and Jason H. Moore (2016). Automating biomedical data science through tree-based pipeline optimization. Applications of Evolutionary Computation, pages 123-137.
# 
# http://rhiever.github.io/tpot/

# In[3]:

def go_tpot():
    from tpot import TPOTRegressor
    import datetime
    tpot = TPOTRegressor(generations=5, population_size=20, verbosity=3, scoring='mean_absolute_error')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('../models/tpot_pipeline_' + datetime.datetime.now().strftime('%Y.%m.%d_%H%M%S')+'.py')


# # Standalone - SGD Regressor

# In[4]:

def go_sgd():
    from sklearn.pipeline import Pipeline
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, make_scorer
    from sklearn.grid_search import GridSearchCV

    scorer = make_scorer(mean_absolute_error)
    clf = linear_model.SGDRegressor()

    # scores = cross_val_score(clf, X, y, cv=10, scoring=scorer)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    param_grid = [{'loss':['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                   'penalty':['none', 'l2', 'l1', 'elasticnet']
                  }]

    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, scoring=scorer)
    gs = gs.fit(X, y)

    print("Best F-Score: ", gs.best_score_)
    print("Best Parameters: ", gs.best_params_)
    print("Best Estimator: ", gs.best_estimator_)


# In[5]:

go_tpot()

