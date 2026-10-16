# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Step 2: Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

print("Dataset Shape:", X.shape)

# Step 3: Create five-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Define machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
}

# Step 5: Perform K-fold cross-validation
results = {}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    results[name] = scores

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i} Validation Score: {score:.4f}")

    mean_score = scores.mean()
    std_score = scores.std()

    print(f"Mean Validation Score: {mean_score:.4f}")
    print(f"Standard Deviation: {std_score:.4f}")

# Step 6: Create the model comparison table
comparison = []

for name, scores in results.items():
    comparison.append([
        name,
        scores[0],
        scores[1],
        scores[2],
        scores[3],
        scores[4],
        scores.mean(),
        scores.std()
    ])

comparison_df = pd.DataFrame(
    comparison,
    columns=[
        "Model",
        "Fold 1",
        "Fold 2",
        "Fold 3",
        "Fold 4",
        "Fold 5",
        "Mean Validation Score",
        "Standard Deviation"
    ]
)

print("\nMODEL COMPARISON")
print(comparison_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

# Step 7: Plot mean validation scores
model_names = list(results.keys())
mean_scores = [results[name].mean() for name in model_names]

plt.figure(figsize=(8, 5))
plt.bar(model_names, mean_scores)
plt.xlabel("Machine Learning Model")
plt.ylabel("Mean Validation R² Score")
plt.title("K-Fold Cross Validation Model Comparison")
plt.ylim(0, 0.6)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
