import pandas as pd
from scipy import stats

# Load cleaned data
df = pd.read_csv("data/StudentsPerformance_clean.csv")

# ANOVA: average score by parental education level
anova_res = stats.f_oneway(
    *[group["average_score"].values for name, group in df.groupby("parental level of education")]
)
print("ANOVA by Parental Education Level:", anova_res)

# Welch t-test: standard vs free/reduced lunch
std_lunch = df[df["lunch"] == "standard"]["average_score"]
fr_lunch = df[df["lunch"] == "free/reduced"]["average_score"]
t_res = stats.ttest_ind(std_lunch, fr_lunch, equal_var=False)
print("Welch t-test (Lunch Type):", t_res)
