# Import, read, and split data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

data = pd.read_csv('data.csv')
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

# Logistic Regression
estimator0 = LogisticRegression(solver='lbfgs')

# Decision Tree
estimator1 = GradientBoostingClassifier()

# Support Vector Machine
estimator2 = SVC(kernel='rbf', gamma=1000)


def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
    return X2, Y2


X2, y2 = randomize(X, y)


def draw_learning_curves(X, y, estimator, num_trainings):

    cv = ShuffleSplit(n_splits=5, test_size=0.3)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=cv, n_jobs=1,
        train_sizes=np.linspace(.1, 1.0, num_trainings))
    print(test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    # plt.plot(train_scores_mean, 'o-', color="g",
    #          label="Training score")
    # plt.plot(test_scores_mean, 'o-', color="y",
    #          label="Cross-validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-Validation Score')
    plt.legend(loc="best")
    plt.show()


draw_learning_curves(X2, y2, estimator2, 10)
cv = ShuffleSplit(n_splits=5, test_size=0.3)

print(cv)
for i, j in cv.split(X2):
    print(i, j)

