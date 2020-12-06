import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score , mean_absolute_error , mean_squared_error
from sklearn.metrics import median_absolute_error , r2_score

veriler= pd.read_excel("verilerim.xlsx","Sayfa1") # Verileri aldığımız exel dosyasını tanımladık. 
veriler.head() # Verileri görüntüledik. 

X = veriler.iloc[:, 3].values # GİRİŞ PARAMETRESİ olarak exel verilerinin 1 ve 3 nolu sütunlarını aldık. 
Y = veriler.iloc[:, 4].values # ÇIKIŞ PARAMETRESİ olarak exel verilerinin 4 nolu sütununu aldık. 
# İloc verisinde virgülden öncesi satırları virgülden sonrası sütunları gösterir .
uzunluk = len(X) 
X = X.reshape((uzunluk,1)) # Giriş parametresini tek sütuna çevirdik. 

print(veriler.isnull().sum()) # Boş parametre olup olmadığını test ettik. 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

from sklearn.linear_model import LinearRegression
model_Regresyon = LinearRegression()
model_Regresyon.fit(X_train,Y_train)
Y_pred = model_Regresyon.predict(X_test)

plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train, model_Regresyon.predict(X_train),color="blue")
plt.title("Sıcaklık Tahmin Değeri")
plt.xlabel("Gate Gerilimi")
plt.ylabel("Sıcaklık")
plt.show()

plt.scatter(X_test,Y_test,color="red")
plt.plot(X_train, model_Regresyon.predict(X_train),color="blue")
plt.title("Sıcaklık Tahmin Değeri")
plt.xlabel("Gate Gerilimi")
plt.ylabel("Sıcaklık")
plt.show()

print("Eğim (Q1) :", model_Regresyon.coef_)
print("Kesen (Q0) :", model_Regresyon.intercept_)
print("y=%0.2f"%model_Regresyon.coef_+"x+%0.2f"%model_Regresyon.intercept_)

#Test veriseti performansı
print("R-Kare (Doğruluk ): ", r2_score(Y_test, Y_pred))
print("MAE :", mean_absolute_error(Y_test, Y_pred))
print("MSE :", mean_squared_error(Y_test, Y_pred))
print("MedAE :", median_absolute_error(Y_test, Y_pred))