import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("Data/train.csv")
os.makedirs("Data/output", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Titanic EDA", fontsize=16)

# Survival count
sns.countplot(x='Survived', data=df, ax=axes[0,0], palette='Set2')
axes[0,0].set_title("Survival Count")

# Survival by class
sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[0,1], palette='Set2')
axes[0,1].set_title("Survival by Class")

# Age distribution
df['Age'].dropna().hist(bins=30, ax=axes[1,0], color='steelblue')
axes[1,0].set_title("Age Distribution")

# Survival by sex
sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[1,1], palette='Set2')
axes[1,1].set_title("Survival by Sex")

plt.tight_layout()
plt.savefig("Data/output/eda_report.png")
print("EDA saved to Data/output/eda_report.png")
plt.show()