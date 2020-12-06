import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import explained_variance_score , mean_absolute_error , mean_squared_error
from sklearn.metrics import median_absolute_error , r2_score

data = pd.read_excel("verilerim.xlsx") # Veriler exel dosyasından okundu. 

print(data.isnull().sum()) # Boş parametre olup olmadığını test ettik. 

x= data.drop('Sicaklik(C)', axis=1) # Sıcaklık Değeri haricindeki verileri aldık. 
#x= x.drop('Zaman(sn)', axis=1) # Sonrasında Silmek istediğimiz başka değer olursa onları da silebiliriz. 
y = data['Sicaklik(C)'] # Sütun ismi Sicaklık(C) olan sütundaki verileri aldık. Çıkış olarak atadık. 

"""
Verileri değişkenlere almak için bu metotda kullanılabilir. 
x = data.iloc[:,:4].values # GİRİŞ PARAMETRESİ olarak exel verilerinin 1 ve 3 nolu sütunlarını aldık. 
y = data.iloc[:, 4].values # ÇIKIŞ PARAMETRESİ olarak exel verilerinin 4 nolu sütununu aldık. 
# İloc verisinde virgülden öncesi satırları virgülden sonrası sütunları gösterir .
"""
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30) # 70% training and 30% test

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators =100 ,max_depth=10 ,random_state =0 ) # Kullanılacak model belirlendi . RF algoritması kullanılacak. 
clf=RandomForestRegressor() # Randomforest regresyonu değişkene atandı. 
clf.fit(X_train,y_train) # Tahminleme yapıldı. 
y_pred=clf.predict(X_test)

feature_names=["Zaman(sn)","Akim(mA)","Kaynak Gerilimi(V)","Gate Gerilimi(V)"] # Parametrelerin Önem Dereceleri Belirlenecek.
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index) # Verilerin önem derecesine göre barplota çizdirilmesi. 
plt.xlabel('Özelliklerin Önem Puanı')
plt.ylabel('Özellikler')
plt.title("Giriş Parametrelerinin Sonucu Etkileme Oranı")
plt.legend()
plt.show()


#scores = cross_val_score(model, X_train, y_train, cv = 10)
#print("Doğruluk :",np.mean(scores)) # Tahminleme puanı için bu yöntem de seçilebilir . 
print("Verilerin sonucu etkileme performansı : ",feature_imp)
print("R-Kare (Doğruluk ): ", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("MedAE :", median_absolute_error(y_test, y_pred))
z=clf.predict([[100,1200,12,4.5]]) # Tahminleme için veriler girdik. Veriler exel sırasına göre olmalı. Kaç adet giriş parametresi varsa o kadar parametre verilmeli. 
print("Tahmini Değer :",z[0])