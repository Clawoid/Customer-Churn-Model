import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

#Loading the dataset into a dataframe
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#Data cleaning and preprocessing
df["TotalCharges"] = df["TotalCharges"].replace(" ", None)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#Remove rows with missing values and reseting the index
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

#Removing the customerID column as it is not useful for prediction
df.drop("customerID", axis=1, inplace=True)

#Converting all categorical variables into numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)
df = df.astype(float)

#Splitting the data into features and target variable
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Training the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Creating predictions and evaluating the model
pred = model.predict(X_test)

#Printing the classification report, confusion matrix, and accuracy score
print("Classification Report:\n", classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Accuracy Score:", accuracy_score(y_test, pred))

pickle.dump(model, open("churn_model.pkl", "wb"))