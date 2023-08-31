from operator import attrgetter

class manual_feature_importance_getter:
    from sklearn.feature_selection import _base

    def __init__(self, estimator, transform_func=None, norm_order=1):
        self.estimator = estimator
        self.transform_func = transform_func
        self.norm_order = norm_order

    def __call__(self, estimator):

        # Get the individual base estimators from the fitted regression chain
        all_estimators = estimator.estimators_
        # importances = np.zeros(shape=(X_train.shape[1], 1))
        # A place to store the feature importances for each estimator
        all_importances = []

        # Iterate through the fitted estimators
        for i in range(len(all_estimators)):

            estimator = all_estimators[i]

            # This is modified from the SKLearn > Feature Selection > _base.py file
            if hasattr(estimator, 'coef_'):
                print('yay!')
                getter = attrgetter('coef_')
                importances = getter(estimator)

            elif hasattr(estimator, 'feature_importances_'):
                print('yay!')
                getter = attrgetter('feature_importances_')
                importances = getter(estimator)

            else:
                print('DOH!')
                importances = np.zeros(shape=(X_train.shape[1], 1))

            # Add the importances to the main list
            all_importances.append(importances)

        # Sum all importances together, and put them into an importances numpy array
        importances = np.array([sum(x) for x in zip(*all_importances)])# 无用

        if self.transform_func is None:
            return importances
        elif self.transform_func == "norm":
            if importances.ndim == 1:
                importances = np.abs(importances)
            else:
                importances = np.linalg.norm(importances, axis=0,
                                             ord=self.norm_order)
        elif self.transform_func == "square":
            if importances.ndim == 1:
                importances = safe_sqr(importances)
            else:
                importances = safe_sqr(importances).sum(axis=0)
        else:
            raise ValueError("Valid values for `self.transform_func` are " +
                             "None, 'norm' and 'square'. Those two " +
                             "transformation are only supported now")

        return importances


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("once")
from sklearn.multioutput import RegressorChain
from sklearn.utils import all_estimators
from sklearn.model_selection import *
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline




# Create multioutput regression dataset and split
X, y = make_regression(n_samples=100, n_features=100, n_informative=25, n_targets=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42)

# Get a list of all SKLearn regressors, and store in dictionary
estimators = all_estimators(type_filter="regressor")
my_estimators = {name[0]: name[1] for name in estimators}

# Iterate through all estimators
for key, value in my_estimators.items():

    # Don't use CV estimators, they hang too often
    if 'cv' in str(key).lower():
        continue

    print(key, value)

    # Get the estimator being used currently
    try:
        estimator = my_estimators[key]()
    except:
        continue

    # Use RFECV on the multiouput regression problem
    # rfe = RFECV(estimator=RegressorChain(base_estimator=estimator, order=[0, 1, 3, 4, 2,
    #                                                                       5]))  # importance_getter=manual_feature_importance_getter(RegressorChain(base_estimator=estimator, order=[0, 1, 3, 4, 2, 5]), None, 1)

    rfe = RFECV(estimator=RegressorChain(base_estimator=estimator, order=[0, 1, 3, 4, 2, 5]),
                importance_getter=manual_feature_importance_getter(
                    RegressorChain(base_estimator=estimator, order=[0, 1, 3, 4, 2, 5]), None, 1))
    model = RegressorChain(base_estimator=estimator, order=[0, 1, 3, 4, 2, 5])
    pipeline = Pipeline(steps=[('s', rfe), ('m', model)])
    # evaluate model
    cv = KFold(n_splits=2)  # n_repeats=2, for Repeated, , random_state=1
    n_scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1,
                               error_score='raise')
    # report performance
    # print('neg_root_mean_squared_error: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))