from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('data.csv')

bmi_life_data = df

bmi_life_model = LinearRegression()

# print(bmi_life_data[['Life expectancy']], bmi_life_data[['BMI']])

bmi_life_model.fit(bmi_life_data[['BMI']],
                   bmi_life_data[['Life expectancy']])

laos_life_exp = bmi_life_model.predict([[21.07931]])

print(laos_life_exp)


