#importing library
import pandas as pd
import numpy as np

#importing data
data=pd.read_csv('Iris.csv')

X= data[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= data['Species']


#test train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


#training the model
lr.fit(X_train,y_train)


#running the model
y_test_1=lr.predict(X_test)



#Evaluating the model
from sklearn.metrics import accuracy_score
print("The accuracy of the model = ",accuracy_score(y_test,y_test_1)*100,'%')


#fill your test cases here 
# Test Case 1 - Iris Setosa
test_case_1 = [5.0, 3.4, 1.5, 0.2]

# Test Case 2 - Iris Versicolor
test_case_2 = [6.1, 2.8, 4.7, 1.2]

# Test Case 3 - Iris Virginica
test_case_3 = [7.9, 3.8, 6.4, 2.0]



prediction1 = lr.predict([test_case_1])
print("Test case 1 prediction = ",prediction1)

prediction2 = lr.predict([test_case_2])
print("Test case 2 prediction = ",prediction2)

prediction3 = lr.predict([test_case_3])
print("Test case 3 prediction = ",prediction3)
