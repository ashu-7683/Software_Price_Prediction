import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("software_prices_large.csv")

# Encode categorical columns
categorical_cols = ["Category", "Platform", "Subscription Type"]
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col in categorical_cols:
    df[col] = label_encoders[col].fit_transform(df[col])

# Feature Engineering: Add "Years Since Release"
df["Years Since Release"] = 2025 - df["Release Year"]

# Define features (X) and target variable (y)
X = df.drop(columns=["Software Name", "Final Price", "Release Year"])
y = df["Final Price"]

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting Model
model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate Model Accuracy
r2 = r2_score(y_test, model.predict(X_test_scaled))
print(f"Model Accuracy (R² Score): {r2:.4f}")

# Get original category names from LabelEncoder
print("Available Categories:", label_encoders["Category"].classes_)
print("Available Platforms:", label_encoders["Platform"].classes_)
print("Available Subscription Types:", label_encoders["Subscription Type"].classes_)


# Function to safely encode user input (handles new/unseen labels)

def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"⚠️ Warning: '{value}' is a new category not seen in training. Assigning 'Unknown'.")
        return len(encoder.classes_)  # Assigning to a new index

print("\nEnter software details to predict price:")

# Get user input
category_names = [str(cat) for cat in df['Category'].unique()]
platform_names = [str(plat) for plat in df['Platform'].unique()]
subscription_names = [str(sub) for sub in df['Subscription Type'].unique()]

# Get user input safely
category = input(f"Enter category ({', '.join(category_names)}): ")
platform = input(f"Enter platform ({', '.join(platform_names)}): ")
subscription = input(f"Enter subscription type ({', '.join(subscription_names)}): ")
features = int(input("Enter number of features: "))
users_supported = int(input("Enter number of users supported: "))
release_year = int(input("Enter release year: "))

# Encode input safely
category_encoded = safe_encode(label_encoders["Category"], category)
platform_encoded = safe_encode(label_encoders["Platform"], platform)
subscription_encoded = safe_encode(label_encoders["Subscription Type"], subscription)
years_since_release = 2025 - release_year

# Ensure input has the same number of features as training data
num_missing_features = X_train.shape[1] - 6  # Find missing features
additional_features = np.zeros(num_missing_features)  # Add zeros for missing features

# Prepare input for prediction with correct shape
user_input = [[category_encoded, platform_encoded, subscription_encoded, features, users_supported, years_since_release] + list(additional_features)]
user_input_scaled = scaler.transform(user_input)

# Predict price
predicted_price = model.predict(user_input_scaled)[0]
print(f"\nPredicted Software Price: ${predicted_price:.2f}")