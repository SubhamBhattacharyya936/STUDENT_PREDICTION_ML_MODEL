# Student Result Prediction using Decision Tree

## Project Overview

This is a beginner-friendly Machine Learning project that predicts whether a student will **Pass** or **Fail** based on:

- Study Hours
- Attendance Percentage
- Assignment Score

The project uses:

- **pandas** for creating and handling data
- **scikit-learn** for splitting data, training a Decision Tree model, and evaluating the model

The model used is a **DecisionTreeClassifier**.

---

## What You Will Learn

By reading and running this project, you will understand:

- How to create a simple dataset using a Python dictionary
- How to convert a dictionary into a pandas DataFrame
- How to separate input features (`X`) and target output (`y`)
- How to split data into training and testing sets
- How to create and train a Decision Tree model
- How to check model accuracy
- How to take user input and make a prediction

---

## Requirements

Install Python 3.8 or above.

Install the required libraries:

```bash
pip install pandas scikit-learn
```

---

## Project Files

```text
student_result_prediction.py   # Main Python code
README.md                      # Project explanation
```

---

## How to Run

1. Save the code as `student_result_prediction.py`.
2. Open a terminal or command prompt in the project folder.
3. Run:

```bash
python student_result_prediction.py
```

4. Enter the requested values:

```text
Enter study hours:
Enter attendance percentage:
Enter assignment score:
```

5. The program will print the predicted result.

---

## Dataset Used

The dataset is created manually inside the code.

| StudyHours | Attendance | Assignment | Result |
|------------|------------|------------|--------|
| 1          | 50         | 40         | Fail   |
| 2          | 55         | 45         | Fail   |
| 2          | 60         | 50         | Fail   |
| 3          | 60         | 50         | Fail   |
| 3          | 65         | 55         | Fail   |
| 4          | 65         | 60         | Fail   |
| 4          | 70         | 60         | Fail   |
| 5          | 70         | 65         | Pass   |
| 5          | 75         | 65         | Pass   |
| 6          | 75         | 70         | Pass   |
| 6          | 80         | 70         | Pass   |
| 7          | 80         | 75         | Pass   |
| 7          | 85         | 80         | Pass   |
| 8          | 85         | 80         | Pass   |
| 8          | 90         | 85         | Pass   |
| 9          | 95         | 90         | Pass   |

---

## Important Note About the Comments

In the original text, some comment lines look like this:

```python
*# -----------------------------*
*# Create Dataset*
*# -----------------------------*
```

This is **not valid Python syntax**. In Python, comments must start with `#`.

Correct version:

```python
# -----------------------------
# Create Dataset
# -----------------------------
```

If you run the code and get `SyntaxError`, remove the `*` symbols from the comment lines.

---

## Corrected Full Code

```python
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
```

---

## Line-by-Line Code Explanation

### 1. Import Libraries

```python
import pandas as pd
```

Imports the pandas library. `pd` is a short alias for pandas. It is used to create and work with DataFrames.

```python
from sklearn.model_selection import train_test_split
```

Imports `train_test_split` from scikit-learn. It is used to divide the dataset into training and testing parts.

```python
from sklearn.tree import DecisionTreeClassifier
```

Imports the Decision Tree classifier. This is the Machine Learning model used for prediction.

```python
from sklearn.metrics import accuracy_score, classification_report
```

Imports evaluation tools:

- `accuracy_score` calculates how many predictions were correct.
- `classification_report` shows precision, recall, F1-score, and support.

---

### 2. Create Dataset

```python
data = {
    "StudyHours": [1, 2, 2, 3, 3, 4, 4, 5,
                   5, 6, 6, 7, 7, 8, 8, 9],
```

Creates a dictionary key called `"StudyHours"`. The list contains study hours for each student.

```python
    "Attendance": [50, 55, 60, 60, 65, 65, 70, 70,
                   75, 75, 80, 80, 85, 85, 90, 95],
```

Creates the `"Attendance"` column. It stores attendance percentages.

```python
    "Assignment": [40, 45, 50, 50, 55, 60, 60, 65,
                   65, 70, 70, 75, 80, 80, 85, 90],
```

Creates the `"Assignment"` column. It stores assignment scores.

```python
    "Result": [
        "Fail", "Fail", "Fail", "Fail",
        "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass"
    ]
}
```

Creates the target column `"Result"`. It contains the final result for each student: `Pass` or `Fail`.

```python
df = pd.DataFrame(data)
```

Converts the dictionary into a pandas DataFrame. A DataFrame is like an Excel table with rows and columns.

---

### 3. Features and Target

```python
X = df[["StudyHours", "Attendance", "Assignment"]]
```

Creates the feature matrix `X`. These are the input columns used to make predictions.

The double brackets `[[ ... ]]` are important because they keep the result as a DataFrame with multiple columns.

```python
y = df["Result"]
```

Creates the target variable `y`. This is the output the model tries to predict.

`y` is a pandas Series because single brackets are used.

---

### 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

Splits the data into training and testing sets.

- `X_train` — features used for training
- `X_test` — features used for testing
- `y_train` — correct answers for training
- `y_test` — correct answers for testing

Parameters:

- `test_size=0.25` means 25% of data is used for testing, and 75% for training.
- `random_state=42` makes the split reproducible. Every run gives the same split.
- `stratify=y` keeps the same ratio of Pass and Fail in both training and testing sets.

---

### 5. Create Model

```python
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

Creates a Decision Tree model.

- `max_depth=3` limits the tree to 3 levels. This keeps the model simple and helps prevent overfitting.
- `random_state=42` makes the model results reproducible.

---

### 6. Train Model

```python
model.fit(X_train, y_train)
```

Trains the model using the training data. The model learns the relationship between study hours, attendance, assignment score, and the final result.

---

### 7. Test Model

```python
y_pred = model.predict(X_test)
```

Uses the trained model to predict results for the test data.

```python
accuracy = accuracy_score(y_test, y_pred)
```

Compares the real test answers `y_test` with the predicted answers `y_pred`.

Accuracy formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

```python
print("===== MODEL EVALUATION =====")
print("Accuracy:", accuracy)
```

Prints the model accuracy.

```python
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))
```

Prints a detailed report containing:

- Precision
- Recall
- F1-score
- Support

These metrics help you understand how well the model performed for each class.

---

### 8. User Input

```python
print("\n===== STUDENT PREDICTION =====")
```

Prints a heading before asking for user input.

```python
study_hours = float(input("Enter study hours: "))
```

Asks the user to enter study hours. `input()` returns text, so `float()` converts it into a decimal number.

```python
attendance = float(input("Enter attendance percentage: "))
```

Asks for attendance percentage and converts it to a number.

```python
assignment = float(input("Enter assignment score: "))
```

Asks for assignment score and converts it to a number.

```python
new_student = [[
    study_hours,
    attendance,
    assignment
]]
```

Creates a new student record.

The model expects a 2D input, so we use double brackets. This represents one row with three columns.

```python
prediction = model.predict(new_student)
```

Uses the trained model to predict the result for the new student.

```python
print("\nPredicted Result:", prediction[0])
```

Prints the predicted result.

`prediction` is an array, so `prediction[0]` gets the first prediction.

---

## Example Output

```text
===== MODEL EVALUATION =====
Accuracy: 1.0

===== CLASSIFICATION REPORT =====
              precision    recall  f1-score   support

        Fail       1.00      1.00      1.00         2
        Pass       1.00      1.00      1.00         2

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4

===== STUDENT PREDICTION =====
Enter study hours: 6
Enter attendance percentage: 78
Enter assignment score: 72

Predicted Result: Pass
```

Your output may be slightly different depending on the input values.

---

## Common Errors and Fixes

### 1. ModuleNotFoundError

If you see:

```text
ModuleNotFoundError: No module named 'pandas'
```

Run:

```bash
pip install pandas scikit-learn
```

### 2. SyntaxError

If you see:

```text
SyntaxError: invalid syntax
```

Check the comment lines. Remove `*` from lines like:

```python
*# -----------------------------*
```

Change them to:

```python
# -----------------------------
```

### 3. ValueError

If you enter text instead of a number, Python may show a `ValueError`.

Enter only numbers, for example:

```text
6
78
72
```

---

## Uploading to GitHub

1. Create a new repository on GitHub.
2. Add your Python file and this `README.md` file.
3. Run these commands in your project folder:

```bash
git init
git add .
git commit -m "Add student result prediction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details.

---

## License

This project is open-source and free to use for learning purposes.


