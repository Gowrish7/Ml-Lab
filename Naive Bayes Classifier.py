#importing packages
import pandas as pd
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


#importing csv file and printing first 5values
data_csv = pd.read_csv("id3.csv")
print("The first 5 values of data\n",data_csv.head())


#printing first 5 training dataset
X=data_csv.iloc[:,:-1]
print("The first 5 value of trained data is\n ",X.head())

Y=data_csv.iloc[:,-1]
print("The first 5 value of trained data is\n ",Y.head())


le_outlook = LabelEncoder()
X.Outlook =le_outlook.fit_transform(X.Outlook)
print(X.head)


X.Temperature =le_outlook.fit_transform(X.Temperature)
print(X.Temperature)

X.Humidity =le_outlook.fit_transform(X.Humidity)
print(X.Humidity)

X.Wind =le_outlook.fit_transform(X.Wind)
print(X.Wind)

le_play = LabelEncoder()
Y.play =le_play.fit_transform(Y)
print("Now the trained output values is\n ",Y.play)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is :", accuracy_score(classifier.predict(X_test),Y_test))
