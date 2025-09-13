import numpy as np
import matplotlib.pyplot as plt

#test data
X = np.array([1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10], dtype=float)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], dtype=float)

m = 0.0 # slope
b = 0.0 # intercept

# Gradient Descent
alpha = 0.01  # learning rate
epochs = 2000 # Number of repetitions
n = len(X)
for _ in range(epochs):
    y_pred = m * X + b
    error = y_pred - y
    dm = (2/n) * np.dot(error, X)
    db = (2/n) * np.sum(error)
    m -= alpha * dm
    b -= alpha * db
print(f"Slope: {m}")
print(f"Intercept: {b}")

plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, m*X + b, color="red", label="Fitted line")
plt.legend()
plt.show()
