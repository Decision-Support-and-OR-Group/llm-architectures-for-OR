import pandas as pd
from scipy.stats import friedmanchisquare

csv_path = "experiment_results.csv"

df = pd.read_csv(csv_path)

conditions = [
    "is_optimal_GPT-5-Mini_1-agent",
    "is_optimal_GPT-5-Mini_2-agent",
    "is_optimal_GPT-5-Mini_3-agent",
    "is_optimal_GPT-5-Mini_4-agent",
    "is_optimal_GPT-5.1_1-agent",
    "is_optimal_GPT-5.1_2-agent",
    "is_optimal_GPT-5.1_3-agent",
    "is_optimal_GPT-5.1_4-agent",
]

data = df[conditions].dropna()

args = [data[col].values for col in conditions]

statistic, p_value = friedmanchisquare(*args)

print("Number of problems used:", len(data))
print("Friedman chi-square statistic:", statistic)
print("p-value:", p_value)


#effect sizes - kendall's w ---
N = len(data)      # number of problems (rows after dropna)
k = len(conditions)  # number of configurations

kendalls_w = statistic / (N * (k - 1))

print(f"Kendall's W (effect size for Friedman): {kendalls_w:.3f}")