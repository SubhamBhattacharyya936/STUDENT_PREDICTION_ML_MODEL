import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Create Dataset
# -----------------------------

data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 4, 5,
                   5, 6, 6, 7, 7, 8, 8, 9],

    "Attendance": [50, 55, 60, 60, 65, 65, 70, 70,
                   75, 75, 80, 80, 85, 85, 90, 95],

    "Assignment": [40, 45, 50, 50, 55, 60, 60, 65,
                   65, 70, 70, 75, 80, 80, 85, 90],

    "Result": [
        "Fail", "Fail", "Fail", "Fail",
        "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Features and Target
# -----------------------------

X = df[["StudyHours", "Attendance", "Assignment"]]
y = df["Result"]

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# Create Model
# -----------------------------

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Test Model
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("===== MODEL EVALUATION =====")
print("Accuracy:", accuracy)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

# -----------------------------
# User Input
# -----------------------------

print("\n===== STUDENT PREDICTION =====")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
assignment = float(input("Enter assignment score: "))

new_student = [[
    study_hours,
    attendance,
    assignment
]]

prediction = model.predict(new_student)

print("\nPredicted Result:", prediction[0])