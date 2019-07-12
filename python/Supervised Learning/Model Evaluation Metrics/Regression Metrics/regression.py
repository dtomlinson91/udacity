from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import tests2 as t
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

boston = load_boston()
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


dec_tree = DecisionTreeRegressor()
ran_for = RandomForestRegressor()
ada = AdaBoostRegressor()
lin_reg = LinearRegression()


dec_tree.fit(X_train, y_train)
ran_for.fit(X_train, y_train)
ada.fit(X_train, y_train)
lin_reg.fit(X_train, y_train)


dec_pred = dec_tree.predict(X_test)
ran_pred = ran_for.predict(X_test)
ada_pred = ada.predict(X_test)
lin_pred = lin_reg.predict(X_test)


# potential model options
a = 'regression'
b = 'classification'
c = 'both regression and classification'

metrics_dict = {
    'precision': b,
    'recall': b,
    'accuracy': b,
    'r2_score': a,
    'mean_squared_error': a,
    'area_under_curve': b,
    'mean_absolute_area': a
}

# checks your answer, no need to change this code
t.q6_check(metrics_dict)
print()

models = {'dec_pred': dec_pred, 'ran_pred': ran_pred, 'ada_pred': ada_pred,
          'lin_pred': lin_pred}
metrics = [r2_score, mean_squared_error, mean_absolute_error]


# Check r2
def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual - preds)**2)
    sst = np.sum((actual - np.mean(actual))**2)
    return 1 - sse / sst


for i in models:
    print(f'r2 manual for {i} is {r2(y_test, models[i]):.4f}')
    print(f'r2 sklearn for {i} is {r2_score(y_test, models[i]):.4f}')
    print()
# Check solution matches sklearn


def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''

    return np.sum((actual - preds)**2) / len(actual)


# Check your solution matches sklearn
for i in models:
    print(f'mse manual for {i} is {mse(y_test, models[i]):.4f}')
    print(f'mse sklearn for {i} is'
          f' {mean_squared_error(y_test, models[i]):.4f}')
    print()


def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''

    return np.sum(np.abs(actual - preds)) / len(actual)


# Check your solution matches sklearn
for i in models:
    print(f'mae manual for {i} is {mae(y_test, models[i]):.4f}')
    print(f'mae sklearn for {i} is'
          f' {mean_absolute_error(y_test, models[i]):.4f}')
    print()

print('=================')
print('Comparison of all models:\n')
for i in models:
    for j in range(len(metrics)):
        print(f'{metrics[j].__name__} for '
              f'{i} {metrics[j](y_test, models[i]):.4f}')
    print()


# match each metric to the model that performed best on it
a = 'decision tree'
b = 'random forest'
c = 'adaptive boosting'
d = 'linear regression'


best_fit = {
    'mse': b,
    'r2': b,
    'mae': b
}

# Tests your answer - don't change this code
t.check_ten(best_fit)
