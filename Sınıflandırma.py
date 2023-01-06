import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('primary_tumor.csv',sep=';')

modelknn = KNeighborsClassifier(n_neighbors=5)

modelnb = GaussianNB()


y = data['class']

x = data.drop('class',axis=1)  

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=1 )

modelknn.fit(X_train,Y_train)

tahminknn = modelknn.score(X_test,Y_test)

print(tahminknn)

modelnb.fit(X_train,Y_train)

tahminnb = modelnb.score(X_test,Y_test)

print(tahminnb)