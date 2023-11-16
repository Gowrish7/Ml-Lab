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



# OUTPUT

1 The first 5 values of data
     Outlook Temperature Humidity    Wind play
0     sunny         hot     high    weak   no
1     sunny         hot     high  strong   no
2  overcast         hot     high    weak  yes
3      rain        mild     high    weak  yes
4      rain        cool   normal    weak  yes

2.The first 5 value of trained data is
      Outlook Temperature Humidity    Wind
0     sunny         hot     high    weak
1     sunny         hot     high  strong
2  overcast         hot     high    weak
3      rain        mild     high    weak
4      rain        cool   normal    weak

3.The first 5 value of trained data is
  0     no
1     no
2    yes
3    yes
4    yes
Name: play, dtype: object


4.le_outlook = LabelEncoder()
X.Outlook =le_outlook.fit_transform(X.Outlook)
print(X.head)
le_outlook = LabelEncoder()
X.Outlook =le_outlook.fit_transform(X.Outlook)
print(X.head)
<bound method NDFrame.head of     Outlook Temperature Humidity    Wind
0         2         hot     high    weak
1         2         hot     high  strong
2         0         hot     high    weak
3         1        mild     high    weak
4         1        cool   normal    weak
5         1        cool   normal  strong
6         0        cool   normal  strong
7         2        mild     high    weak
8         2        cool   normal    weak
9         1        mild   normal    weak
10        2        mild   normal  strong
11        0        mild     high  strong
12        0         hot   normal    weak
13        1        mild     high  strong>


5.0     1
1     1
2     1
3     2
4     0
5     0
6     0
7     2
8     0
9     2
10    2
11    2
12    1
13    2
Name: Temperature, dtype: int32

6.0     0
1     0
2     0
3     0
4     1
5     1
6     1
7     0
8     1
9     1
10    1
11    0
12    1
13    0
Name: Humidity, dtype: int32

7.0     1
1     0
2     1
3     1
4     1
5     0
6     0
7     1
8     1
9     1
10    0
11    0
12    1
13    0
Name: Wind, dtype: int32

8.Now the trained output values is
  [0 0 1 1 1 0 1 0 1 1 1 1 1 0]

9.GaussianNB()

10.Accuracy is : 0.6666666666666666



