import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Data/train.csv")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

model = joblib.load("models/model.pkl")

scores = cross_val_score(model, X, y, cv=5)
print("=== 5-Fold Cross Validation ===")
print(f"Scores : {scores.round(3)}")
print(f"Mean   : {scores.mean():.3f}")
print(f"Std Dev: {scores.std():.3f}")

y_pred = model.predict(X)
print("\n=== Classification Report ===")
print(classification_report(y, y_pred, target_names=['Not Survived', 'Survived']))

print("=== Confusion Matrix ===")
print(confusion_matrix(y, y_pred))