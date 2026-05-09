from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

def create_pipeline():

    numeric_features = ["Pclass", "Age", "Fare"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    return pipeline