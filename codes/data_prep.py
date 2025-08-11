import pandas as pd

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")
df.columns = [c.strip() for c in df.columns]

# Calculate average score
df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# Save cleaned dataset
df.to_csv("data/StudentsPerformance_clean.csv", index=False)

print("Data preparation complete. Saved as data/StudentsPerformance_clean.csv")
