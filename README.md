# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# Download from: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
df = pd.read_csv("titanic.csv")

# 2. Select relevant features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df[features + ["Survived"]]

# 3. Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 4. Encode categorical variables
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# 5. Split features (X) and target (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Example prediction
sample_passenger = pd.DataFrame({
    "Pclass": [1],
    "Age": [29],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [100],
    "Sex_male": [0],       # 0 = female, 1 = male
    "Embarked_Q": [0],
    "Embarked_S": [1]
})
prediction = model.predict(sample_passenger)[0]
print("Survival Prediction:", "Survived" if prediction == 1 else "Did not survive")
