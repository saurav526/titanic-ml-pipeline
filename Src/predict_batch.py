import pandas as pd
import joblib
import os

model = joblib.load("models/model.pkl")

df = pd.read_csv("Data/test.csv")

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
df['Prediction'] = model.predict(df[features])
df['Survived_Label'] = df['Prediction'].map({0: 'No', 1: 'Yes'})

os.makedirs("Data/output", exist_ok=True)
df.to_csv("Data/output/predictions.csv", index=False)
print(f"Done! {len(df)} predictions saved to Data/output/predictions.csv")