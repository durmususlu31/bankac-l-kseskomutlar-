#bankacilik sesli komutlar-
banka sektöründe müşteri hizmetlerinde müşteri hizmetlerinin kullanıcının isteğine göre kategori yapıp istek alanına aktarması
# Metin Sınıflandırma (Text Classification)

Bu proje, Türkçe metinleri **Random Forest Classifier** kullanarak sınıflandırmak için hazırlanmıştır.  
Amaç, kullanıcıdan alınan bir metnin hangi kategoriye ait olduğunu tahmin etmektir.  

## 🚀 Kullanılan Teknolojiler
- Python
- pandas, numpy
- scikit-learn (CountVectorizer, RandomForestClassifier)


---

## 📜 main.py (Temizlenmiş Kod)
```python
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Stopwords listesi
stopwords = [
    'fakat','lakin','ancak','acaba','ama','aslında','az','bazı','belki',
    'biri','birkaç','birşey','biz','bu','çok','çünkü','da','daha','de',
    'defa','diye','eğer','en','gibi','hem','hep','hepsi','her','hiç',
    'için','ile','ise','kez','ki','kim','mı','mu','mü','nasıl','ne',
    'neden','nerde','nerede','nereye','niçin','niye','o','sanki','şey',
    'siz','şu','tüm','ve','veya','ya','yani'
]

# ------------------------------
# 1. Veri Setini Yükle
# ------------------------------
df = pd.read_csv("data/yenibanka.csv")

# ------------------------------
# 2. Kullanıcıdan Mesaj Al
# ------------------------------
mesaj = input("Yapmak İstediniz İşlemi Giriniz: ")
mesajdf = pd.DataFrame({"metin": [mesaj], "kategori": [0]})
df = pd.concat([df, mesajdf], ignore_index=True)

# ------------------------------
# 3. Ön İşleme (Stopwords Temizleme, Lowercase)
# ------------------------------
def temizle_metin(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # noktalama işaretlerini kaldır
    text = re.sub(r'\d+', ' ', text)      # sayıları kaldır
    text = re.sub(r'\s+', ' ', text)      # fazla boşlukları temizle
    # stopwords kaldır
    text = re.sub(r'\b(?:' + '|'.join(stopwords) + r')\b', ' ', text)
    return text.strip()

df["metin"] = df["metin"].astype(str).apply(temizle_metin)

# ------------------------------
# 4. Vektörleştirme
# ------------------------------
cv = CountVectorizer(max_features=500)
x = cv.fit_transform(df["metin"]).toarray()
y = df["kategori"]

# Kullanıcı mesajını ayır
tahmin = x[-1].copy()
x = x[:-1]
y = y[:-1]

# ------------------------------
# 5. Eğitim ve Tahmin
# ------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=21
)

model = RandomForestClassifier(n_estimators=300, random_state=21)
model.fit(x_train, y_train)

skor = model.score(x_test, y_test)
sonuc = model.predict([tahmin])

# ------------------------------
# 6. Çıktı
# ------------------------------
print("Sonuç:", sonuc, "Skor:", skor)
