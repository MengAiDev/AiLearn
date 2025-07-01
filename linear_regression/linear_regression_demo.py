# Linear Regression Demo
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load diabetes dataset (real-life example)
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
# Evaluate model performance
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error: {:.2f}".format(mean_squared_error(y_test, y_pred)))
print("R² Score: {:.2f}".format(r2_score(y_test, y_pred)))

# Show model equation for the first feature as example
print("\nModel equation for first feature: y = {:.2f} + {:.2f}x0".format(
    model.intercept_, model.coef_[0]))
print("Note: This dataset has {} features".format(X.shape[1]))

