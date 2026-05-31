import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

le_type = LabelEncoder()
le_rating = LabelEncoder()
le_director = LabelEncoder()

df = pd.read_csv("netflix_titles.csv")

df["director"] = df["director"].fillna("Unknown")
df["country"] = df["country"].fillna("Not Available")
df["cast"] = df["cast"].fillna("No Cast")

#print(df[["type", "rating", "release_year"]].isnull().sum())

df_ml = df[["type", "rating", "release_year", "director"]].dropna()

#Encoding
df_ml["type"] = le_type.fit_transform(df_ml["type"])
df_ml["rating"] = le_rating.fit_transform(df_ml["rating"])
df_ml["director"] = le_director.fit_transform(df_ml["director"])

#Features 
X = df_ml[["rating", "release_year", "director"]]

#Target
Y = df_ml["type"]

#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42
)

#Decision Tree Model
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)
print(dt_pred[:10])
print("DT Accuracy = ", accuracy_score(Y_test, dt_pred))
print("DT Confusion Matrix = \n", confusion_matrix(Y_test, dt_pred))

#Random Forest Model
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
print(rf_pred[:10])
print("RF Accuracy = ", accuracy_score(Y_test, rf_pred))
print("RF Confusion Matrix = \n", confusion_matrix(Y_test, rf_pred))

#Logistic regression Model
lr = LogisticRegression(max_iter = 1000)
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)
print(lr_pred[:10])
print("LR Accuracy = ", accuracy_score(Y_test, lr_pred))
print("LR Confusion Matrix = \n", confusion_matrix(Y_test, lr_pred))
