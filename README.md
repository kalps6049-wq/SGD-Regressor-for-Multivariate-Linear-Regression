# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Start.
2. 2.Data Preparation.

3.Hypothesis Definition.

4.Cost Function.

5.Parameter Update Rule.

6.Iterative Training.

7.Model Evaluation.

8.End.
 

## Program:
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()

X = data.data[:, :3]
Y = np.c_[data.target, data.data[:, 6]]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MultiOutputRegressor(SGDRegressor(random_state=42))
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nSample Predictions (House Price, Population):")
print(Y_pred[:5])

DEVELOPED BY:KALPANA M
REGISTER NO:212225240064
```

## Output:
<img width="1225" height="205" alt="Screenshot 2026-04-28 095722" src="https://github.com/user-attachments/assets/cdc4156b-e984-4dbe-81fd-18a86280f0c5" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
