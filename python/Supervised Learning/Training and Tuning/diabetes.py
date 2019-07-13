# Import our libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
import sys
import os
sys.path.append(os.getcwd())
import check_file as ch
import warnings
warnings.filterwarnings("ignore")


sns.set(style="ticks")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)

# Read in our dataset
diabetes = pd.read_csv('diabetes.csv')

# Take a look at the first few rows of the dataset
print(diabetes.head())

# Summary on diabetes df
print(diabetes.describe())
sns.pairplot(diabetes, hue='Outcome', diag_kind='hist')
plt.show()
sns.heatmap(diabetes.corr(), annot=True, square=True,
            annot_kws={"size": 12})
# plt.tight_layout()
plt.show()

diabetes.hist()
plt.show()

preg = diabetes.groupby(['Pregnancies', 'Outcome'])\
    .agg({'Pregnancies': 'count'})
preg = preg.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))


# Set our testing and training data
y = diabetes['Outcome']
X = diabetes.drop(['Outcome'], axis=1)


AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=200, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
# build a classifier
clf_rf = RandomForestClassifier()

# Set up the hyperparameter search
param_dist = {"max_depth": [3, None],
              "n_estimators": list(range(10, 200)),
              "max_features": list(range(1, X_test.shape[1] + 1)),
              "min_samples_split": list(range(2, 11)),
              "min_samples_leaf": list(range(1, 11)),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
rf_preds = random_search.best_estimator_.predict(X_test)

ch.print_metrics(y_test, rf_preds, 'random forest')

# Print the parameters used in the model
print(random_search.best_estimator_)


# build a classifier for ada boost
clf_ada = AdaBoostClassifier()

# Set up the hyperparameter search
# look at  setting up your search for n_estimators, learning_rate
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
param_dist = {"n_estimators": [10, 100, 200, 400],
              "learning_rate": [0.001, 0.005, .01, 0.05, 0.1, 0.2, 0.3, 0.4,
                                0.5, 1, 2, 10, 20]}

# Run a randomized search over the hyperparameters
ada_search = RandomizedSearchCV(clf_ada, param_distributions=param_dist)

# Fit the model on the training data
ada_search.fit(X_train, y_train)

# Make predictions on the test data
ada_preds = ada_search.best_estimator_.predict(X_test)

# Return your metrics on test data
ch.print_metrics(y_test, ada_preds, 'adaboost')

# Print the hyperparams used
print(ada_search.best_estimator_)

# Doing the same as above except using a full GridSearch

ada_grid_search = GridSearchCV(clf_ada, param_dist)

ada_grid_search.fit(X_train, y_train)

ada_grid_preds = ada_grid_search.best_estimator_.predict(X_test)

ch.print_metrics(y_test, ada_grid_preds, 'adaboost')

print(ada_grid_search.best_estimator_)


# build a classifier for support vector machines
clf_svc = SVC()

# Set up the hyperparameter search
# look at setting up your search for C (recommend 0-10 range),
# kernel, and degree
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
param_dist = {"C": [0.1, 0.5, 1, 3, 5],
              "kernel": ['linear', 'rbf']
              }


# Run a randomized search over the hyperparameters
svc_search = RandomizedSearchCV(clf_svc, param_distributions=param_dist)

# Fit the model on the training data
svc_search.fit(X_train, y_train)

# Make predictions on the test data
svc_preds = svc_search.best_estimator_.predict(X_test)

ch.print_metrics(y_test, svc_preds, 'svc')

print(svc_search.best_estimator_)


# Get information about the best model
# Get column names
features = diabetes.columns[:diabetes.shape[1]]
print(features)

# Get the best features of the best estimator (random forest)
importances = ada_search.best_estimator_.feature_importances_
print(importances)

# Get these in increasing number of importance
indices = np.argsort(importances)
print(indices)

# Plot them
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
