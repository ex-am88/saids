# Step 1: Import statistical functions
import numpy as np
from scipy.stats import f_oneway, kruskal

print("Libraries imported successfully.")

# Step 2: Define the sample marks of three groups of students
group1 = [78, 82, 85, 90, 88]
group2 = [65, 70, 72, 68, 66]
group3 = [90, 92, 95, 94, 91]

print("Group 1:", group1)
print("Group 2:", group2)
print("Group 3:", group3)

# Step 3: Perform One-Way ANOVA to compare group means
f_stat, p_value_anova = f_oneway(group1, group2, group3)

print("===== ONE-WAY ANOVA =====")
print("F Statistic:", round(f_stat, 4))
print("P Value:", f"{p_value_anova:.6f}")

if p_value_anova < 0.05:
    print("Reject Null Hypothesis")
    print("Means are significantly different")
else:
    print("Fail to Reject Null Hypothesis")
    print("Insufficient evidence of a difference in means")
    
    # Step 4: Perform the non-parametric Kruskal-Wallis test
h_stat, p_value_kruskal = kruskal(group1, group2, group3)

print("===== KRUSKAL-WALLIS TEST =====")
print("H Statistic:", round(h_stat, 4))
print("P Value:", f"{p_value_kruskal:.6f}")

if p_value_kruskal < 0.05:
    print("Reject Null Hypothesis")
    print("Groups differ significantly")
else:
    print("Fail to Reject Null Hypothesis")
    print("Insufficient evidence of a difference between groups")