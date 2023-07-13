import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import re
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load the stopwords
stop_words = stopwords.words('english')

# Load the news dataset
news_dataset = pd.read_csv('C:/Users/LENOVO/Documents/Fake_News_Detection/train.csv')
news_dataset = news_dataset.fillna('')

# Create a new column 'content' by combining 'author' and 'title'
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Prepare the input features (X) and target variable (Y)
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Perform stemming and preprocessing on the text data
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Extract features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, Y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_content = request.form['content']
    stemmed_content = stemming(news_content)
    input_data = vectorizer.transform([stemmed_content])
    prediction = model.predict(input_data)[0]

    # Generate the confusion matrix
    train_predictions = model.predict(X)
    cm = confusion_matrix(Y, train_predictions)

    # Generate the learning curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Fake', 'Real'])
    plt.yticks([0, 1], ['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Save the confusion matrix plot to a buffer
    confusion_matrix_buffer = io.BytesIO()
    plt.savefig(confusion_matrix_buffer, format='png')
    confusion_matrix_buffer.seek(0)

    # Convert the buffer to a base64 encoded string
    confusion_matrix_image = base64.b64encode(confusion_matrix_buffer.read()).decode('utf-8')

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-Validation Score')
    plt.legend(loc='best')

    # Save the learning curve plot to a buffer
    learning_curve_buffer = io.BytesIO()
    plt.savefig(learning_curve_buffer, format='png')
    learning_curve_buffer.seek(0)

    # Convert the buffer to a base64 encoded string
    learning_curve_image = base64.b64encode(learning_curve_buffer.read()).decode('utf-8')

    return render_template('index.html', prediction=prediction,
                           confusion_matrix_image=confusion_matrix_image,
                           learning_curve_image=learning_curve_image)

if __name__ == '__main__':
    app.run(debug=True)
