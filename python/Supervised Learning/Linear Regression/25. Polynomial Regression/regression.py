import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

df = pd.read_csv('data.csv')
# print(df)
X = df[['Var_X']]
y = df[['Var_Y']]

poly_feat = PolynomialFeatures(degree=2)

X_poly = poly_feat.fit_transform(X)

poly_model = LinearRegression(fit_intercept=False).fit(X_poly, y)
print(poly_model)
# sns.lineplot(x='Var_X', y='Var_Y', data=df)
# plt.show()

