import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('data.csv', header=None)

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# Create the standardization scaling object
scaler = StandardScaler()

# Scale and fit the standardization paramaeters
X_scaled = scaler.fit_transform(X)

# Create the LR model with Lasso regularization
lasso_reg = Lasso()

# Fit the model
lasso_reg.fit(X_scaled, y)

# Get the regression coeficients
reg_coef = lasso_reg.coef_
print(reg_coef)
