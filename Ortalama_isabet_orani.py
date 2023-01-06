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

toplamknnoran = 0

toplamnboran = 0

iterasyon = 1000

for x1 in range(1000,iterasyon+1000):  

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=x1 )

    modelknn.fit(X_train,Y_train)

    tahminknn = modelknn.score(X_test,Y_test)
    
    toplamknnoran += tahminknn


    modelnb.fit(X_train,Y_train)

    tahminnb = modelnb.score(X_test,Y_test)
    
    toplamnboran += tahminnb
    
print(toplamknnoran/iterasyon)

print(toplamnboran/iterasyon)
