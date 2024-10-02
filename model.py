import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('The_Cancer_data_1500_V2.csv')

print(df.head())

target_column = 'Diagnosis'  

X = df.drop(columns=[target_column]) 
y = df[target_column]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred))

model_filename = 'cancer_prediction_model.joblib'
joblib.dump(model, model_filename)

print(f"Model saved as {model_filename}")