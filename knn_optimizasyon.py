import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


data = pd.read_csv('primary_tumor.csv',sep=';')

modelknn = KNeighborsClassifier(n_neighbors=5)

y = data['class']

x = data.drop('class',axis=1)  

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=1 )

cozumler = np.empty((0, 15), dtype=bool)

for x1 in range(2):
    for x2 in range(2):
         for x3 in range(2):
             for x4 in range(2):
                 for x5 in range(2):
                     for x6 in range(2):
                         for x7 in range(2):
                             for x8 in range(2):
                                 for x9 in range(2):
                                     for x10 in range(2):
                                         for x11 in range(2):
                                             for x12 in range(2):
                                                 for x13 in range(2):
                                                     for x14 in range(2):
                                                         for x15 in range(2):
                                                             cozum = np.array([bool(x1),bool(x2),bool(x3),bool(x4),bool(x5),bool(x6),bool(x7),bool(x8),bool(x9),bool(x10),bool(x11),bool(x12),bool(x13),bool(x14),bool(x15)])
                                                             cozumler = np.vstack((cozumler,cozum))
                                                             
print("cozumler olusturuldu")
             
modelknn.fit(X_train,Y_train)

tahminknn = modelknn.score(X_test,Y_test)

print("özellik seçimi olmadan isabet oranı: ",tahminknn)


mevcut_satir = np.random.randint(1,32767)

print("secilen ilk çözüm: ",mevcut_satir)

modelknn.fit(X_train.loc[:,cozumler[mevcut_satir]],Y_train)

mevcut_skor = modelknn.score(X_test.loc[:,cozumler[mevcut_satir]],Y_test)

while True:
    
    ust_satir = mevcut_satir-1
    ust_skor = 0
    
    alt_satir = mevcut_satir+1
    alt_skor = 0
    
    if ust_satir > 0:
        modelknn.fit(X_train.loc[:,cozumler[ust_satir]],Y_train)
        ust_skor = modelknn.score(X_test.loc[:,cozumler[ust_satir]],Y_test)
        
    if alt_satir < 32768:
        modelknn.fit(X_train.loc[:,cozumler[alt_satir]],Y_train)
        alt_skor = modelknn.score(X_test.loc[:,cozumler[alt_satir]],Y_test)
    
    if alt_skor <= mevcut_skor and ust_skor <= mevcut_skor:
        break
        
    if alt_skor <= mevcut_skor and ust_skor > mevcut_skor:
        mevcut_satir = ust_satir
        mevcut_skor = ust_skor
        
    if alt_skor > mevcut_skor and ust_skor <= mevcut_skor:    
        mevcut_satir = alt_satir
        mevcut_skor = alt_skor
    
    if alt_skor > mevcut_skor and ust_skor > mevcut_skor:
        if alt_skor > ust_skor:
            mevcut_satir = alt_satir
            mevcut_skor = alt_skor
        else:
            mevcut_satir = ust_satir
            mevcut_skor = ust_skor

print("bulunan en iyi çözüm: ",mevcut_satir)
print("çözümün isabet oranı: ",mevcut_skor)
print("çözüm satırı: ",cozumler[mevcut_satir])














