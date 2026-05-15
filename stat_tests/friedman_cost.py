import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

csv_path = "experiment_results.csv"

df = pd.read_csv(csv_path)

conditions = [
    "cost_usd_GPT-5-Mini_1-agent",
    "cost_usd_GPT-5-Mini_2-agent",
    "cost_usd_GPT-5-Mini_3-agent",
    "cost_usd_GPT-5-Mini_4-agent",
    "cost_usd_GPT-5.1_1-agent",
    "cost_usd_GPT-5.1_2-agent",
    "cost_usd_GPT-5.1_3-agent",
    "cost_usd_GPT-5.1_4-agent",
]

data = df[conditions].dropna()

args = [data[col].values for col in conditions]


statistic, p_value = friedmanchisquare(*args)

print("Number of problems used:", len(data))
print("Friedman chi-square statistic:", statistic)
print("p-value:", p_value)

#effect sizes - kendall's w ---
N = len(data)     
k = len(conditions)  

kendalls_w = statistic / (N * (k - 1))

print(f"Kendall's W (effect size for Friedman): {kendalls_w:.3f}")

# -------- Nemenyi post-hoc test --------
values = data.to_numpy()

nemenyi_results = sp.posthoc_nemenyi_friedman(values)

nemenyi_results.index = conditions
nemenyi_results.columns = conditions

print("\nNemenyi post-hoc p-value matrix:")
print(nemenyi_results)

# -------- Nemenyi post-hoc nicer print --------
alpha = 0.05
significant_table = nemenyi_results < alpha

print(f"\nSignificant at alpha = {alpha} (True = p < {alpha}):")
print(significant_table)


# Compute median  per configuration
medians = data.median(axis=0) 

print("\nMedian  per configuration:")
for cond, med in medians.items():
    print(f"{cond}: {med:.3f}")

# Boolean matrix: is row significantly different than column?

cheaper_than = pd.DataFrame(False, index=conditions, columns=conditions)

for i, ci in enumerate(conditions):
    for j, cj in enumerate(conditions):
        if i == j:
            continue
        p = nemenyi_results.loc[ci, cj]
        if p < alpha and medians[ci] < medians[cj]:
            cheaper_than.loc[ci, cj] = True

print(f"\nIs row significantly smaller (p < {alpha}) than column?")
print(cheaper_than)


