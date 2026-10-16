# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Define the original sample
data = np.array([65, 70, 72, 68, 75, 80, 74, 69, 77, 71])

# Step 3: Calculate the original sample mean
sample_mean = np.mean(data)
print("Original Sample:", data)
print("Sample Mean:", sample_mean)

# Step 4: Set the number of bootstrap samples
n_bootstrap = 10000
bootstrap_means = []

# Step 5: Perform bootstrap resampling
for i in range(n_bootstrap):
    bootstrap_sample = np.random.choice(
        data,
        size=len(data),
        replace=True
    )
    bootstrap_mean = np.mean(bootstrap_sample)
    bootstrap_means.append(bootstrap_mean)

bootstrap_means = np.array(bootstrap_means)

# Step 6: Calculate bootstrap standard error
standard_error = np.std(bootstrap_means, ddof=1)

# Step 7: Calculate the 95% confidence interval
lower = np.percentile(bootstrap_means, 2.5)
upper = np.percentile(bootstrap_means, 97.5)

print("\nBootstrap Results")
print("-----------------")
print("Bootstrap Standard Error:", round(standard_error, 4))
print("95% Confidence Interval:", (round(lower, 4), round(upper, 4)))

# Step 8: Plot the bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=40)

plt.axvline(sample_mean, linestyle="--", linewidth=2, label="Original Mean")
plt.axvline(lower, linestyle="--", linewidth=2, label="Lower 95% CI")
plt.axvline(upper, linestyle="--", linewidth=2, label="Upper 95% CI")

plt.xlabel("Bootstrap Sample Mean")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Sample Mean")
plt.legend()
plt.show()

