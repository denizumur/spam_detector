
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')
df = pd.read_csv("spam_ham_dataset.csv")  # CSV dosyanı buraya eklediğinden emin ol

print(df.head())

# Veri Temizleme 
def temizle(text):
    if isinstance(text, str):  # Boş değerleri önlemek için
        text = text.lower()  # Küçük harfe çevir
        text = re.sub(r"\d+", "", text)  # Sayıları kaldır
        text = text.translate(str.maketrans("", "", string.punctuation))  # Noktalama işaretlerini kaldır
        words = word_tokenize(text)  # Kelimelere ayır
        words = [word for word in words if word not in stopwords.words("english")]  # Stopwords temizleme
        return " ".join(words)
    return ""

df["clean_text"] = df["text"].apply(temizle)

X = df["clean_text"]
y = df["label_num"]  # 0 = Ham (Normal), 1 = Spam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF ile Özellik Çıkarımı
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Unigram ve Bigram kullan
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Modeli Eğitme (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


print(f"Modelin Doğruluk Oranı: {accuracy_score(y_test, y_pred):.2f}")
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Eposta Girisi
while True:
    kullanici_email = input("\ Bir e-posta giriniz (çıkmak için 'cikis' yaz): ")

    if kullanici_email.lower() == "cikis":
        print("cikis yaptiniz")
        break

    temiz_email = temizle(kullanici_email)  # Temizleme işlemi
    email_tfidf = vectorizer.transform([temiz_email])  # TF-IDF vektörüne çevir

    tahmin = model.predict(email_tfidf)[0]  # Modelin tahmini

    if tahmin == 1:
        print(" **DİKKAT SPAM olabilir!!!!**")
    else:
        print(" **GÜVENLİ görünüyor.**")
