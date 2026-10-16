# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Original dataset shape:", X.shape)

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce four features to two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Create a DataFrame for transformed data
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Species"] = [target_names[i] for i in y]

# Step 6: Display explained variance
print("\nExplained Variance Ratio:")
print("PC1:", round(pca.explained_variance_ratio_[0], 4))
print("PC2:", round(pca.explained_variance_ratio_[1], 4))

total_variance = pca.explained_variance_ratio_.sum()
print("Total Variance Retained:", round(total_variance, 4))

# Step 7: Display first ten transformed observations
print("\nFirst 10 Transformed Observations:")
print(pca_df.head(10).round(4))

# Step 8: Visualize transformed feature space
plt.figure(figsize=(9, 6))

for i, species in enumerate(target_names):
    indices = y == i
    plt.scatter(
        X_pca[indices, 0],
        X_pca[indices, 1],
        label=species,
        alpha=0.75
    )

plt.xlabel(
    f"Principal Component 1 "
    f"({pca.explained_variance_ratio_[0] * 100:.2f}% variance)"
)
plt.ylabel(
    f"Principal Component 2 "
    f"({pca.explained_variance_ratio_[1] * 100:.2f}% variance)"
)
plt.title("PCA: Transformed Feature Space of Iris Dataset")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()