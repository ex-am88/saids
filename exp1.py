# Step 1: Import the required Python libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

print("Libraries imported successfully.")

# Step 2: Load the Iris dataset
iris = load_iris()

print("Iris dataset loaded successfully.")

# Step 3: Create a Pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species column
df["Species"] = iris.target
df["Species"] = df["Species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("DataFrame created successfully.")

# Step 4: Display the first five records
print(df.head())

# Step 5: Display column names, data types, and non-null counts
df.info()

# Step 6: Display the number of rows and columns
print("Shape:", df.shape)

# Step 7: Count missing values in each column
print(df.isnull().sum())

# Step 8: Display count, mean, standard deviation, minimum, maximum, and quartiles
print(df.describe())

# Step 9: Visualize the distribution of numerical features
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

# Step 10: Identify the spread and potential outliers in numerical features
plt.figure(figsize=(8, 5))
sns.boxplot(data=df.iloc[:, :4])
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Step 11: Visualize relationships between features for each species
sns.pairplot(df, hue="Species")
plt.show()

# Step 12: Display correlations between numerical features
plt.figure(figsize=(6, 5))
sns.heatmap(
    df.iloc[:, :4].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.tight_layout()
plt.show()