# K-Nearest Neighbors on Diabetes Dataset

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

# 1. Load dataset (kept in the dataset folder beside this script)
DATA_PATH = Path(__file__).resolve().parent / "dataset" / "diabetes.csv"
df = pd.read_csv(DATA_PATH)

# Display first 5 records
print("First 5 rows:")
print(df.head())

# 2. Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Feature scaling
# KNN uses distance, so scaling is important
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# 6. Train the model
knn.fit(X_train, y_train)

# 7. Make predictions
y_pred = knn.predict(X_test)

# 8. Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 9. Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 10. Display results
print("\n----- KNN RESULTS -----")

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

print("\nError Rate:", error_rate)
print("Error Rate (%):", error_rate * 100)

print("\nPrecision:", precision)

print("Recall:", recall)

# -----------------------------------------------------------------------------
# EXPLANATION (added):
# 1. This program uses K-Nearest Neighbors (KNN) to classify records in the
#    diabetes dataset. Outcome is the target: 1 or 0.
# 2. The other columns are input features. train_test_split reserves 20% of
#    records for testing and stratify preserves the class proportions.
# 3. StandardScaler standardizes features so a variable with large numeric
#    values does not dominate the distance calculation. It is fitted only on
#    the training data, then applied to the test data.
# 4. KNeighborsClassifier(n_neighbors=5) predicts a test record using the
#    majority class among its five nearest training records.
# 5. The confusion matrix, accuracy, error rate, precision, and recall measure
#    the classifier's performance on unseen test records.
#
# RUN ON LINUX:
#   cd sem7_annotated
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
#   python3 ml/K_Nearest_Neighbors.py
# Dataset required: ml/dataset/diabetes.csv (download_datasets.sh can fetch it).
# -----------------------------------------------------------------------------
