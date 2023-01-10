import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

#binary_repr int lerin string biçiminde binary karşılıklarını döndürüyor

def int2boolarray(sayi):
    cozum = np.array([],dtype=bool)
    binarysayi = np.binary_repr(sayi,15)
    for x in range(len(binarysayi)):
        deger = bool(int(binarysayi[0]))
        cozum = np.append(cozum, deger)
        binarysayi = binarysayi[1:]
    return cozum
    
        

data = pd.read_csv('primary_tumor.csv',sep=';')

modelknn = KNeighborsClassifier(n_neighbors=5)

y = data['class']

x = data.drop('class',axis=1)  

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=1 )
                        
             
modelknn.fit(X_train,Y_train)

tahminknn = modelknn.score(X_test,Y_test)

print("özellik seçimi olmadan isabet oranı: ",tahminknn)



randomcombination = np.random.randint(1,32767)

eniyi_cozum = int2boolarray(randomcombination)

print("secilen ilk çözüm: ",randomcombination)

modelknn.fit(X_train.loc[:,eniyi_cozum],Y_train)

eniyi_skor = modelknn.score(X_test.loc[:,eniyi_cozum],Y_test)

crntcmbine = randomcombination

while True:
    b = crntcmbine-1
    c = crntcmbine+1
    komsucozumler = np.array([b,c])
    
    cozum_ayni_mi = True
    
    for y in range(2):
        
        cozum = int2boolarray(komsucozumler[y])
        
        modelknn.fit(X_train.loc[:,cozum],Y_train)

        skor = modelknn.score(X_test.loc[:,cozum],Y_test)
        
        if(skor>eniyi_skor):
            eniyi_skor = skor
            crntcmbine = komsucozumler[y]
            eniyi_cozum = cozum
            cozum_ayni_mi = False
        
    if(cozum_ayni_mi):
        break

print("bulunan en iyi çözüm: ",crntcmbine)
print("çözümün isabet oranı: ",eniyi_skor)
print("seçilen özellikler: ",eniyi_cozum)














