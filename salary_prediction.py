# STEP 1: Upload ZIP file
from google.colab import files
uploaded = files.upload()  # Upload the archive.zip or your file

# STEP 2: Extract ZIP
import zipfile
import os

zip_file = list(uploaded.keys())[0]  # Get uploaded zip file name
extract_dir = "extracted"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

# STEP 3: Load the CSV
import pandas as pd

csv_path = os.path.join(extract_dir, "ai_job_dataset.csv")
df = pd.read_csv(csv_path)

# STEP 4: Clean and Prepare Data
df = df.drop(columns=['job_id', 'required_skills', 'posting_date', 'application_deadline', 'company_name'])
df.dropna(inplace=True)

# STEP 5: Define Features and Target
target = 'salary_usd'
X = df.drop(columns=[target])
y = df[target]

# Categorical and Numerical Columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# STEP 6: Preprocessing Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# STEP 7: Full Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# STEP 8: Train-Test Split and Model Fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# STEP 9: Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f" Mean Squared Error: {mse:.2f}")
print(f" R² Score: {r2:.4f}")  # Should be ~0.87
print(f" Prediction Score: {model.score(X_test, y_test):.4f}")

# STEP 10: Save the Model
joblib.dump(model, "salary_prediction_model.pkl")
print(" Model saved as salary_prediction_model.pkl")