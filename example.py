from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
import pandas as pd
from scipy.stats import loguniform


#  Grid search takes 2 arguments
# 1. the model of whoch you want to optimize the hyperparameters
# 2. the search space

...
# define model
model = LogisticRegression()
# define search space
space = dict()


search = GridSearchCV(model, space)


# Gridearch provides an cv argument where you can handover the cnumbe rof folds or an crossvalidation object.
# It also provides a scoring parmeter which takes a string indicating the metric to optimize.
# “n_jobs” argument as an integer with the number of cores in your system, e.g. 8. Or you can set it to be -1 to automatically use all of the cores in your system.

# Once this is defined one can call the fit() method to train and evaluate model hyperparameter combinations using cross-validation.
# At the end of the search, you can access all of the results via attributes on the class. Perhaps the most important attributes are the best score observed and the hyperparameters that achieved the best score.

# summarize the sonar dataset
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
dataframe = pd.read_csv(url, header=None)
# split into input and output elements
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
print(X.shape, y.shape)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

...
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)

search = RandomizedSearchCV(
    model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)

...
# execute search
result = search.fit(X, y)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
