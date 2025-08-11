import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load cleaned data
df = pd.read_csv("data/StudentsPerformance_clean.csv")

# Figure 1: Histogram of average scores
plt.figure()
plt.hist(df["average_score"], bins=20, edgecolor="black")
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("picture/Fig1_Distribution_Avg_Scores.png", dpi=300)
plt.close()

# Figure 2: Boxplot by parental education
plt.figure()
df.boxplot(column="average_score", by="parental level of education", rot=45)
plt.title("Average Score by Parental Education Level")
plt.suptitle("")
plt.xlabel("Parental Level of Education")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("picture/Fig2_Parental_Ed_Boxplot.png", dpi=300)
plt.close()


# Figure 3: Boxplot by lunch type
plt.figure()
df.boxplot(column="average_score", by="lunch")
plt.title("Average Score by Lunch Type")
plt.suptitle("")
plt.xlabel("Lunch Type")
plt.ylabel("Average Score")
plt.tight_layout()
plt.savefig("picture/Fig3_LunchType_Boxplot.png", dpi=300)
plt.close()

# Figure 4: Top 10 Random Forest feature importances
X = df.drop(columns=["average_score"])
y = df["average_score"]
X_enc = pd.get_dummies(X, drop_first=False)
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_enc, y)

importances = pd.Series(rf.feature_importances_, index=X_enc.columns)
imp_top10 = importances.sort_values(ascending=False).head(10)

plt.figure()
ypos = np.arange(len(imp_top10))[::-1]
plt.barh(ypos, imp_top10.values)
plt.yticks(ypos, imp_top10.index)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("picture/Fig4_Feature_Importance_Updated.png", dpi=300)
plt.close()

print("Figures saved to picture/")
