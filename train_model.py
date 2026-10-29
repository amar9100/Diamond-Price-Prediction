import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Load dataset
df = pd.read_csv("diamonds.csv")

# Step 2: Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Step 3: Define features and target
X = df.drop(columns=["price"])
y = df["price"]

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Step 5: Identify feature types
numerical_features = [
    "carat", "depth", "table", "x", "y", "z"
]

categorical_features = [
    "cut", "color", "clarity"
]

# Step 6: Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Step 7: Create the complete KNN pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "knn",
            KNeighborsRegressor(n_neighbors=5)
        )
    ]
)

# Step 8: Train the model
model.fit(X_train, y_train)

print("KNN model training completed successfully!")

# Step 9: Predict prices for the test data
y_pred = model.predict(X_test)

# Step 10: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

print("\n----- MODEL EVALUATION -----")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Step 11: Save the trained model using Pickle
with open("diamond_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as diamond_model.pkl")