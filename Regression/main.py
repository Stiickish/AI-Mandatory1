#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

cars_df = pd.read_csv('../Data/cars.csv')
cars_df_copy = cars_df.copy()
cars_df_copy['horsepower'] = pd.to_numeric(cars_df_copy['horsepower'], errors='coerce')
mean_horsepower = round(cars_df_copy['horsepower'].mean(), 1)
cars_df_copy['horsepower'].fillna(mean_horsepower, inplace=True)
cars_df_copy.to_csv('../Data/cars_cleaned.csv', index=False)

print(cars_df_copy)

#%%
#LinearRegression
X = cars_df_copy[['horsepower', 'weight']]
y = cars_df_copy[['acceleration']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


model = LinearRegression()

model.fit(X_train,y_train)

print("Cofficients:", model.coef_)
print("Intercept", model.intercept_)
#%%
#Messure Error for LinearRegression
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test,y_pred)

print(f'Mean Squared Error (MSE: {mse})')
print(f'R-squared (R^2): {r2})')



#%%
#Poly Regression
X = cars_df_copy[['horsepower', 'weight']]
y = cars_df_copy['acceleration']  # Assuming 'acceleration' is a one-dimensional array

# Split your dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a PolynomialFeatures transformer and a LinearRegression model
# Replace 'degree' with the degree of polynomial you want to test
polynomial_features = PolynomialFeatures(degree=4, include_bias=False)
linear_regression = LinearRegression()

# Create a pipeline that first transforms the features and then fits the model
model = Pipeline([("polynomial_features", polynomial_features),
                  ("linear_regression", linear_regression)])

# Fit the model on the training data
model.fit(X_train, y_train)

# Get the coefficients and intercept from the pipeline's linear regression step
coefficients = model.named_steps['linear_regression'].coef_
intercept = model.named_steps['linear_regression'].intercept_

# Print the coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

#%%

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R²)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

#%%

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R^2): {r2}')

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 5))

# Scatter plot of actual vs. predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Acceleration')
plt.ylabel('Predicted Acceleration')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

# Residuals plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='red')
plt.title('Residuals Plot')
plt.xlabel('Predicted Acceleration')
plt.ylabel('Residuals')
plt.axhline(y=0, color='green', linestyle='--')

plt.tight_layout()
plt.show()
# %%