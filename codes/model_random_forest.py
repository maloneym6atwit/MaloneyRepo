import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load cleaned data
df = pd.read_csv("data/StudentsPerformance_clean.csv")

X = df.drop(columns=["average_score"])
y = df["average_score"]

# One-hot encode categoricals
X_enc = pd.get_dummies(X, drop_first=False)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_enc, y)

# Save importances
importances = pd.Series(rf.feature_importances_, index=X_enc.columns)
importances.sort_values(ascending=False).to_csv("results/rf_feature_importances.csv")

print("Random Forest complete. Importances saved to results/rf_feature_importances.csv")
