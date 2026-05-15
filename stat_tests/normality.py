import pandas as pd
from scipy.stats import anderson

csv_path = "experiment_results.csv"

df = pd.read_csv(csv_path)

# columns to check
cols = [
    "cost_usd_GPT-5-Mini_1-agent",
    "cost_usd_GPT-5-Mini_2-agent",
    "cost_usd_GPT-5-Mini_3-agent",
    "cost_usd_GPT-5-Mini_4-agent",
    "cost_usd_GPT-5.1_1-agent",
    "cost_usd_GPT-5.1_2-agent",
    "cost_usd_GPT-5.1_3-agent",
    "cost_usd_GPT-5.1_4-agent",
    "latency_sec_GPT-5-Mini_1-agent",
    "latency_sec_GPT-5-Mini_2-agent",
    "latency_sec_GPT-5-Mini_3-agent",
    "latency_sec_GPT-5-Mini_4-agent",
    "latency_sec_GPT-5.1_1-agent",
    "latency_sec_GPT-5.1_2-agent",
    "latency_sec_GPT-5.1_3-agent",
    "latency_sec_GPT-5.1_4-agent",
    "is_optimal_GPT-5-Mini_1-agent",
    "is_optimal_GPT-5-Mini_2-agent",
    "is_optimal_GPT-5-Mini_3-agent",
    "is_optimal_GPT-5-Mini_4-agent",
    "is_optimal_GPT-5.1_1-agent",
    "is_optimal_GPT-5.1_2-agent",
    "is_optimal_GPT-5.1_3-agent",
    "is_optimal_GPT-5.1_4-agent",
]

print("Anderson–Darling test for normality (SciPy):\n")

results = []

for col in cols:
    x = df[col].dropna().values  # remove NaNs
    
    # Anderson–Darling test for normal distribution
    res = anderson(x, dist='norm')
    stat = res.statistic
    crit = res.critical_values
    sig_levels = res.significance_level
    
    # Decide at alpha = 0.05 (5%)
    alpha = 5.0
    if alpha in sig_levels:
        idx = list(sig_levels).index(alpha)
        crit_5 = crit[idx]
        reject_5 = stat > crit_5
    else:
        crit_5 = float("nan")
        reject_5 = False

    results.append((col, stat, crit_5, reject_5))

    print(f"Column: {col}")
    print(f"  A^2 statistic      : {stat:.4f}")
    print(f"  Critical value (5%): {crit_5:.4f}")
    print(f"  Reject normality at 5%? {'YES' if reject_5 else 'NO'}")
    print(f"  All critical values: {crit}")
    print(f"  Significance levels: {sig_levels}\n")

# Summary table
summary = pd.DataFrame(results, columns=["column", "A2_stat", "crit_5", "reject_normal_5pct"])
print("\nSummary (alpha = 0.05):")
print(summary)