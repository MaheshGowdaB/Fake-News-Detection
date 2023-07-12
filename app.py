from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load the stopwords
stop_words = stopwords.words('english')

# Load the news dataset
news_dataset = pd.read_csv('C:/Users/LENOVO/Documents/Fake News Detection/train.csv')
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

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predict the labels for the test set
Y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Generate confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(cm)

# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, Y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate the mean and standard deviation of training and testing scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('learning_curve.png')

# Save the confusion matrix plot
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_content = request.form['content']
    stemmed_content = stemming(news_content)
    input_data = vectorizer.transform([stemmed_content])
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        prediction_text = 'Real'
    else:
        prediction_text = 'Fake'
    return render_template('index.html', prediction=prediction_text, confusion_matrix='confusion_matrix.png', learning_curve='learning_curve.png', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
