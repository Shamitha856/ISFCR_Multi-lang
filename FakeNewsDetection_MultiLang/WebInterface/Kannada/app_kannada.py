import numpy as np
from sklearn import *
from flask import Flask, render_template, request
from flask_cors import CORS
import os
import flask 
import pickle
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
from textblob import TextBlob
import langid


app = Flask(__name__)
vectorization = TfidfVectorizer()
loaded_model = pickle.load(open('model_kannada_pac.pkl', 'rb'))
dataframe = pd.read_csv('final_df_kannada.csv')
x = dataframe['text_name']
y = dataframe['real/fake']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    xv_train = vectorization.fit_transform(x_train.values.astype('U'))
    xv_test = vectorization.transform(x_test.values.astype('U'))
    input_data = [news]
    vectorized_input_data = vectorization.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__,static_url_path='', static_folder="static", template_folder="templates")

app.config['DEBUG'] = True

# App routing is used to map the specific URL with the associated function that is intended to perform some task. 
@app.route('/')
def main():
    return render_template("index_kannada.html")

     
@app.route('/predict_text', methods=['POST'])
# A function to predict the text based news implemented using passive aggressive classifier
def predict_text():
    if request.method == 'POST':
        message = request.form['news']
        language = langid.classify(message)
        #language = detect_langs(message)
        #print(lang.detect_language())
        print(language)
        language=language[0]
        if message.startswith("https://" or "http://" or "www."):
            print('URL')
            pred2=['Url']
            return render_template('index_kannada.html', prediction2=pred2)
        elif language == 'en' or language == 'hi':
            print("En or Hn")
            pred2 = ['enORhn_text']
            return render_template('index_kannada.html', prediction2=pred2) 
        else:
            pred2 = fake_news_det(message)
            print(pred2)
            return render_template('index_kannada.html', prediction2=pred2)        
    else:
        return render_template('index_kannada.html', prediction2=pred2)
    
    

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(port= port, debug=True, use_reloader=False)