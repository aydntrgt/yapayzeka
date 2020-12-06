import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

veriler= pd.read_excel("verilerim.xlsx","Sayfa1") # Verileri aldığımız exel dosyasını tanımladık. 
veriler.head() # Verileri görüntüledik. 

print(veriler.isnull().sum()) # Boş parametre olup olmadığını test ettik. 

X = veriler.iloc[:,[1,3]].values # GİRİŞ PARAMETRESİ olarak exel verilerinin 1 ve 3 nolu sütunlarını aldık. 
Y = veriler.iloc[:, 4].values # ÇIKIŞ PARAMETRESİ olarak exel verilerinin 4 nolu sütununu aldık. 
# İloc verisinde virgülden öncesi satırları virgülden sonrası sütunları gösterir .

