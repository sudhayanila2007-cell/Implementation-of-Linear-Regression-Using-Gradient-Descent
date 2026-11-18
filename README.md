# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the 50 Startups dataset.
2. Encode the categorical variable (State) into numerical values.
3. Separate the input features and target variable (Profit).
4. Split the dataset into training and testing data and standardize them.
5. Train the model using Gradient Descent by minimizing the cost function.
6. Predict the profit, evaluate the model, and plot the cost convergence graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Udhayanila S
RegisterNumber: 212225040476 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Upload dataset
from google.colab import files
uploaded = files.upload()

# Load dataset
data = pd.read_csv(next(iter(uploaded)))

print("First 5 rows of the dataset:")
display(data.head())

# Encode categorical column
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

print("\nEncoded dataset:")
display(data_encoded.head())

# Separate features and target
X_raw = data_encoded.drop('Profit', axis=1).values
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)

# Split dataset into training and testing
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Standardize data
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train_raw)
X_test = X_scaler.transform(X_test_raw)

y_train = y_scaler.fit_transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

# Add bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, num_iters=5000, tol=1e-9):
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y

        gradient = (1 / m) * X.T.dot(errors)
        theta -= learning_rate * gradient

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            print(f"Converged at iteration {i}")
            break

        prev_cost = cost

    return theta, J_history

# Train model
theta, J_hist = gradient_descent(
    X_train, y_train,
    learning_rate=0.01,
    num_iters=5000
)

print("\nLearned Parameters (Theta):")
print(theta.flatten())

# Plot cost convergence
plt.figure(figsize=(7, 4))
plt.plot(J_hist)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.show()

# Predict test data
y_pred_scaled = X_test.dot(theta)

# Convert predictions to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)

# Evaluate model
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test_raw, y_pred))
print("R2 Score:", r2_score(y_test_raw, y_pred))
```

## Output:
<img width="1325" height="644" alt="image" src="https://github.com/user-attachments/assets/32287a08-d6c7-4b01-8972-bf278ba9caff" />
<img width="1326" height="642" alt="image" src="https://github.com/user-attachments/assets/0f54d03a-5685-476c-903a-97ada808c63b" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
