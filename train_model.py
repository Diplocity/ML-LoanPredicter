import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("data/train_loandata.csv")
df = df.drop("Loan_ID", axis=1)

# -----------------------------
# Handle missing values
# -----------------------------

categorical_cols = df.select_dtypes(include="object").columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# -----------------------------
# Encode categorical data
# -----------------------------

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Features & target
# -----------------------------

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model (Random Forest)
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("MODEL ACCURACY:", round(accuracy * 100, 2), "%")

# -----------------------------
# Save model + encoders
# -----------------------------

joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("Model saved successfully 🚀")