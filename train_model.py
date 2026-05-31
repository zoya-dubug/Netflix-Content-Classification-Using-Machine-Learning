import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("netflix_titles.csv")

df["director"] = df["director"].fillna("Unknown")
df["country"] = df["country"].fillna("Not Available")
df["cast"] = df["cast"].fillna("No Cast")

#print(df[["type", "rating", "release_year"]].isnull().sum())

# ==========================
# Data Loading
# ==========================
df_ml = df[["type", "rating", "release_year", "director"]].dropna()
print("ML Dataset Shape: ", df_ml.shape)
print("\nFeatures Used: ")
print(["rating", "release_year", "director"])

# ==========================
# Data Preprocessing
# ==========================
le_type = LabelEncoder()
le_rating = LabelEncoder()
le_director = LabelEncoder()
df_ml["type"] = le_type.fit_transform(df_ml["type"])
df_ml["rating"] = le_rating.fit_transform(df_ml["rating"])
df_ml["director"] = le_director.fit_transform(df_ml["director"])

#Features 
X = df_ml[["rating", "release_year", "director"]]

#Target
y = df_ml["type"]

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# ==========================
# Model Training & Evaluation
# ==========================
#Decision Tree Model
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("\nDecision Tree Results: ")
print("Accuracy = ", accuracy_score(y_test, dt_pred))
print("Confusion Matrix = \n", confusion_matrix(y_test, dt_pred))

#Random Forest Model
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("\nRandom Forest Results: ")
print("Accuracy = ", accuracy_score(y_test, rf_pred))
print("Confusion Matrix = \n", confusion_matrix(y_test, rf_pred))

#Logistic regression Model
lr = LogisticRegression(max_iter = 1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("\nLogistic Regression Results: ")
print("Accuracy = ", accuracy_score(y_test, lr_pred))
print("Confusion Matrix = \n", confusion_matrix(y_test, lr_pred))

# ==========================
# Comparision Table
# ==========================
results = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression"
    ],
    "Accuracy": [
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, lr_pred)
    ]
})

print("\nModel Comparison")
print(results.sort_values("Accuracy", ascending = False))
