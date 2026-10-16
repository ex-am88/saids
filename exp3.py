# Step 1: Import statistical tests from SciPy
from scipy.stats import f_oneway, chi2_contingency

alpha = 0.05

print("Libraries imported successfully.")
print("Significance level:", alpha)

# Step 2: Define marks of students from three classes
group1 = [68, 72, 65, 70, 75, 71, 69, 74]
group2 = [78, 80, 76, 82, 79, 81, 77, 84]
group3 = [60, 64, 62, 65, 61, 63, 66, 67]

print("Group 1:", group1)
print("Group 2:", group2)
print("Group 3:", group3)

# Step 3: Perform One-Way ANOVA on the three groups
f_stat, p_value_anova = f_oneway(group1, group2, group3)

print("===== ONE-WAY ANOVA =====")
print("F-Statistic:", round(f_stat, 4))
print("P-Value:", f"{p_value_anova:.6f}")

if p_value_anova < alpha:
    print("Decision: Reject Null Hypothesis")
    print("Conclusion: At least one group mean is significantly different.")
else:
    print("Decision: Fail to Reject Null Hypothesis")
    print("Conclusion: Insufficient evidence of a difference among group means.")
    
# Step 4: Create the contingency table for the Chi-Square test
table = [
    [30, 20],
    [15, 35]
]

print("Contingency Table:")
print(table)

# Step 5: Perform the Chi-Square test of independence
chi2, p_value_chi, dof, expected = chi2_contingency(table)

print("===== CHI-SQUARE TEST =====")
print("Chi-Square Statistic:", round(chi2, 4))
print("Degrees of Freedom:", dof)
print("P-Value:", round(p_value_chi, 6))

print("\nExpected Frequencies:")
for row in expected:
    print([round(value, 2) for value in row])

if p_value_chi < alpha:
    print("\nDecision: Reject Null Hypothesis")
    print("Conclusion: There is evidence of an association between the variables.")
else:
    print("\nDecision: Fail to Reject Null Hypothesis")
    print("Conclusion: Insufficient evidence of an association between the variables.")