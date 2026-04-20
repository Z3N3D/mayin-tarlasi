import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Load the data
csv_path = 'landmine_tabular_data.csv'
print("Loading data...")
df = pd.read_csv(csv_path)

# 5 features to use
features = ['area', 'circularity', 'mean_intensity', 'thermal_contrast', 'edge_density']
target = 'label'

print(f"Total number of records: {len(df)}")

# Split the train and test sets
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

if len(train_df) == 0 or len(test_df) == 0:
    print("'split' column not found, splitting manually...")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Create and train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the results
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model_filename = 'random_forest_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as '{model_filename}'.")

# Predictions and probabilities for the first 5 records
print("\nMine probabilities for the first 5 records from the test set:")
probabilities = model.predict_proba(X_test.head(5))
predictions = model.predict(X_test.head(5))

for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    print(f"Sample {i+1}: Probability of being a Mine: {prob[1]*100:.2f}% -> Prediction: {'Mine (1)' if pred == 1 else 'Not a Mine (0)'}")