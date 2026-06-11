from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION SYSTEM")
print("=" * 50)

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("\nDataset Information")
print("-" * 30)
print("Total Records:", len(X))
print("Features:", X.shape[1])
print("Classes:", len(set(y)))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Create Model
model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Performance")
print("-" * 30)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report")
print("-" * 30)
print(classification_report(y_test, predictions))

# Confusion Matrix
print("Confusion Matrix")
print("-" * 30)
print(confusion_matrix(y_test, predictions))

print("\nModel Evaluation Completed Successfully!")