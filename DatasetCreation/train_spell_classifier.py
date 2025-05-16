import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Load saved training data ===
X = np.load("X_spells.npy")  # shape: (num_samples, 784)
y = np.load("y_spells.npy")  # labels: 0 = open, 1 = close

# === Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === Define a pipeline with scaling and classifier ===
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

# === Define a hyperparameter grid for tuning ===
param_grid = {
    'clf__kernel': ['rbf', 'poly', 'sigmoid'],
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
    'clf__degree': [2, 3, 4]  # Only used for 'poly' kernel
}

# === Perform grid search ===
print("Performing hyperparameter tuning with GridSearchCV...")
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# === Evaluate performance ===
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Parameters: {grid.best_params_}")
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Save the best model ===
joblib.dump(grid.best_estimator_, "new_custom_classifier.pkl", compress=3)
print("Model saved as O_P_classifier.pkl")
