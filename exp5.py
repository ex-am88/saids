# Step 1: Import NumPy, Pearson correlation, and Linear Regression
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# Step 2: Define study hours as X and exam marks as Y
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([35, 40, 50, 55, 65, 70, 75, 85, 90, 95])

print("Study Hours (X):")
print(X.flatten())

print("\nExam Marks (Y):")
print(Y)


# Step 3: Calculate Pearson's correlation coefficient and p-value
r, p = pearsonr(X.flatten(), Y)

print("===== CORRELATION =====")
print("Correlation Coefficient (r):", round(r, 4))
print("P-value:", f"{p:.6f}")

if p < 0.05:
    print("Significant Correlation Exists")
else:
    print("No Significant Correlation")
    
# Step 4: Fit a Linear Regression model to the study-hours data
model = LinearRegression()
model.fit(X, Y)

print("Linear Regression model trained successfully.")

# Step 5: Display the regression equation parameters
slope = model.coef_[0]
intercept = model.intercept_

print("===== LINEAR REGRESSION =====")
print("Slope:", round(slope, 4))
print("Intercept:", round(intercept, 4))

print("\nRegression Equation:")
print(f"Marks = {slope:.4f} × Study Hours + {intercept:.4f}")

# Step 6: Predict exam marks for 11 study hours
predicted = model.predict([[11]])

print("Predicted Marks for 11 Study Hours:", round(predicted[0], 2))

# Step 7: Evaluate how well the model fits the given data
score = model.score(X, Y)

print("R² Score:", round(score, 4))

# Step 8: Plot the actual marks and the predicted regression line
plt.figure(figsize=(8, 5))

plt.scatter(X.flatten(), Y, label="Actual Marks")
plt.plot(X.flatten(), model.predict(X), label="Regression Line")

plt.title("Study Hours vs Exam Marks")
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.legend()
plt.grid(True)
plt.show()