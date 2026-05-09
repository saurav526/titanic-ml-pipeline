import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(model_path)

new_data = pd.DataFrame({
    "Pclass": [3],
    "Age": [22],
    "Fare": [7.25]
})

prediction = model.predict(new_data)

print("Prediction (0 = No, 1 = Yes):", prediction[0])