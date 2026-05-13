# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

df = pd.read_csv('Placement_Data.csv')
df1 = df.copy()
df1 = df1.drop(['sl_no', 'salary'], axis=1)
le = LabelEncoder()
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
for col in categorical_cols:
    df1[col] = le.fit_transform(df1[col])
x = df1.iloc[:, :-1]
y = df1['status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['true', 'false'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
```
## Output:
<img width="548" height="346" alt="Screenshot 2026-05-13 112252" src="https://github.com/user-attachments/assets/63d611cf-7124-4782-a6e4-59b3489d635a" />


<img width="828" height="553" alt="Screenshot 2026-05-13 112113" src="https://github.com/user-attachments/assets/c599d5f5-61e2-4bb2-8d48-9388793ab12e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
