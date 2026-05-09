import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from pipeline import create_pipeline

# so basically we made the file loading os indepandent !!
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw", "titanic.csv")

df = pd.read_csv(data_path)

X = df[["Pclass", "Age", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = create_pipeline()

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print("Model Accuracy:", accuracy)

joblib.dump(pipeline, "models/model.pkl")
print("Model saved successfully.")