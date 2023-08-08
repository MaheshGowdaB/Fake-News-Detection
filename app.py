import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

import nltk
nltk.download('stopwords')

app = Flask(__name__)

stop_words = stopwords.words('english')

# Load the trained model and vectorizer
news_dataset = pd.read_csv('C:/Users/LENOVO/Documents/Fake_News_Detection/train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

X = news_dataset['content'].values
Y = news_dataset['label'].values

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
#Logistic Regression Model
model = LogisticRegression()
model.fit(X, Y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        news_text_stemmed = stemming(news_text)
        news_vector = vectorizer.transform([news_text_stemmed])
        prediction = model.predict(news_vector)[0]
        if prediction == 1:
            return render_template('index.html', prediction="REAL")
        else:
            return render_template('index.html', prediction="FAKE")

if __name__ == '__main__':
    app.run(debug=True)  #run
