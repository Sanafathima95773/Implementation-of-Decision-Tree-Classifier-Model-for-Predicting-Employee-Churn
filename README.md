# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```1.import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.
```

## Program:
```
/* Program to implement the Decision Tree Classifier Model for Predicting Employee Churn
Developed by : Sana Fathima H
Register no: 212223240145
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/a215305a-66f5-4aa6-ab5b-0245d2746345)
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/c303e936-17cb-4cad-a488-188653617e6e)
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/3549e736-d771-4c54-91b5-94290454b8ea)
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/98f99fa3-72de-4413-8f6c-7b25a6b565b7)
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/48787cc6-c6eb-4dbb-bda4-50014b8a0114)
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/8bbf7d58-798f-4703-8a18-a59c7ed24ca9)
![image](https://github.com/Sanafathima95773/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147084627/8dbcab36-3b04-49d4-856c-dc90fb5cc665)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
