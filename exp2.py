# Step 1: Import NumPy, Matplotlib, and Seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully.")

# Step 2: Generate random samples for three probability distributions
normal_data = np.random.normal(loc=50, scale=10, size=1000)
binomial_data = np.random.binomial(n=10, p=0.5, size=1000)
poisson_data = np.random.poisson(lam=5, size=1000)

print("Random samples generated successfully.")

# Step 3: Visualize the Normal Distribution
plt.figure(figsize=(6, 4))
sns.histplot(normal_data, kde=True)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Step 4: Visualize the Binomial Distribution
plt.figure(figsize=(6, 4))
sns.histplot(binomial_data, bins=np.arange(-0.5, 11.5, 1))
plt.title("Binomial Distribution")
plt.xlabel("Number of Successes")
plt.ylabel("Frequency")
plt.xticks(range(11))
plt.show()

# Step 5: Visualize the Poisson Distribution
plt.figure(figsize=(6, 4))
sns.histplot(poisson_data, discrete=True)
plt.title("Poisson Distribution")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.show()

