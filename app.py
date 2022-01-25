from flask import Flask, request
import nltk
from nltk.stem import WordNetLemmatizer
import re
import pickle

app = Flask(__name__)

word = WordNetLemmatizer()


def preprocess(data):
    a = re.sub('[^a-zA-Z]', ' ', data)
    a = a.lower()
    a = a.split()
    a = [word.lemmatize(testWord) for testWord in a]
    a = ' '.join(a)
    return a


count_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('mdetectModel.pkl', 'rb'))


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form['mood_pred']
    a = preprocess(msg)

    result = model.predict(count_vectorizer.transform([a]))[0]
    resultstr = str(result)
    return resultstr


if __name__ == '__main__':
    app.run(debug=True)
    app.run()
