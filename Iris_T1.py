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


