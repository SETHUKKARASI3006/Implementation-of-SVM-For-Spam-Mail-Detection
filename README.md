# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing: Load data, check for nulls, and split into training/testing sets.

2. Feature Extraction: Use CountVectorizer to transform text data into numerical format.

3. Model Training: Train the SVM model (SVC) using the training data.

4. Model Evaluation: Predict using the test set and calculate accuracy score.


## Program and Output:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```
<br>

```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
```
<br>

```
data.head()
```
<br>

![o1](/1.png)
<br>

```
data.tail()
```
<br>

![o2](/2.png)
<br>

```
data.info()
```
<br>

![o3](/3.png)
<br>

```
data.isnull().sum()
```
<br>

![o4](/4.png)
<br>

```
x=data['v2'].values
y=data['v1'].values
```
<br>

```
x.shape
```
<br>

![o5](/5.png)
<br>

```
y.shape
```
<br>

![o6](/6.png)
<br>

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
```
<br>

```
x_train.shape
```
<br>

![o7](/7.png)
<br>

```
x_test.shape
```
<br>

![o8](/8.png)
<br>

```
y_train.shape
```
<br>

![o9](/9.png)
<br>

```
y_test.shape
```
<br>

![o10](/10.png)
<br>

```
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
<br>

```
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
<br>

```
x_train.shape
```
<br>

![o11](/11.png)
<br>

```
type(x_train)
```
<br>

![o12](/12.png)
<br>

```
x_test.shape
```
<br>

![o13](/13.png)
<br>

```
type(x_test)
```
<br>

![o14](/14.png)
<br>

```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
<br>

![o15](/15.png)
<br>

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
<br>

![o16](/16.png)
<br>

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
