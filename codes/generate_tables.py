import pandas as pd

df = pd.read_csv("data/StudentsPerformance_clean.csv")

# Table 1: Average score by parental education
table1 = df.groupby("parental level of education")["average_score"].mean()
table1.to_csv("results/table_parental_education.csv")

# Table 2: Average score by lunch type
table2 = df.groupby("lunch")["average_score"].mean()
table2.to_csv("results/table_lunch.csv")

# Table 3: Average score by test preparation course
table3 = df.groupby("test preparation course")["average_score"].mean()
table3.to_csv("results/table_test_prep.csv")

# Save basic descriptive stats
stats_summary = df.describe()
stats_summary.to_csv("results/stats_summary.csv")

print("Tables saved to results/")
