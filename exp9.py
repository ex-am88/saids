# Step 1: Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.stats import ttest_rel

# Step 2: Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

print("Dataset shape:", X.shape)

# Step 3: Create five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Define machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ])
}

# Step 5: Perform cross-validation
results = {}

for name, model in models.items():
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
    )

    rmse = -cv_results["test_rmse"]
    r2 = cv_results["test_r2"]

    results[name] = {"rmse": rmse, "r2": r2}

# Step 6: Display fold-wise results
for name, result in results.items():
    print("\n", name)
    print("RMSE:", np.round(result["rmse"], 3))
    print("R2:", np.round(result["r2"], 3))

# Step 7: Calculate mean and standard deviation
summary = []

for name, result in results.items():
    summary.append([
        name,
        result["rmse"].mean(),
        result["rmse"].std(),
        result["r2"].mean(),
        result["r2"].std()
    ])

summary_df = pd.DataFrame(
    summary,
    columns=["Model", "Mean RMSE", "RMSE Std", "Mean R2", "R2 Std"]
)

print("\nModel Comparison:")
print(summary_df.round(4).to_string(index=False))

# Step 8: Perform paired statistical comparisons
model_names = list(models.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1 = model_names[i]
        model2 = model_names[j]

        t_stat, p_value = ttest_rel(
            results[model1]["rmse"],
            results[model2]["rmse"]
        )

        print("\nStatistical Comparison:")
        print(model1, "vs", model2)
        print("t-statistic:", round(t_stat, 4))
        print("p-value:", round(p_value, 4))

        if p_value < 0.05:
            print("Result: Significant difference")
        else:
            print("Result: No significant difference")
