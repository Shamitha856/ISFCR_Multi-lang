import numpy as np
from sklearn import *
# Flask is a micro web framework written in Python. It does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.
# render_template is used to generate output from a template file.
# “CORS” stands for Cross-Origin Resource Sharing. It allows you to make requests from one website to another website in the browser, which is normally prohibited by another browser policy called the Same-Origin Policy (SOP).
from flask import Flask, render_template, request
from flask_cors import CORS
import newspaper
# The OS module in Python provides a way of using operating system dependent functionality. The functions that the OS module provides allows you to interface with the underlying operating system that Python is running on – be that Windows, Mac or Linux
import os
# When the Flask application handles a request, it creates a Request object based on the environment it received from the WSGI server. Because a worker (thread, process, or coroutine depending on the server) handles only one request at a time, the request data can be considered global to that worker during that request.
import flask
# Pickle in Python is primarily used in serializing and deserializing a Python object structure. In other words, it's the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.
import pickle
# Newspaper is a python library for extracting & curating articles.
from newspaper import Article
# Urllib module is the URL handling module for python. It is used to fetch URLs (Uniform Resource Locators).
import urllib
# Natural Language Toolkit(NLTK) is a leading platform for building Python programs to work with human language data. 
import nltk
# Punkt Sentence Tokenizer. This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences. It must be trained on a large collection of plaintext in the target language before it can be used.
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from django.core.validators import URLValidator 
from django.core.exceptions import  ValidationError
import requests
# Python-tesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images.
import pytesseract
# Python Imaging Library. This library can be used to manipulate images quite easy.
from PIL import Image
from flask_uploads import UploadSet, configure_uploads, IMAGES
from googletrans import Translator
import googletrans
from textblob import TextBlob
import langid

app = Flask(__name__)
vectorization = TfidfVectorizer(stop_words='english',max_df=0.7)
# To retrieve pickled data, we use pickle.load() function to do that. The primary argument of pickle load function is the file object that you get by opening the file in read-binary (rb) mode.
loaded_model = pickle.load(open('model_english_pac.pkl', 'rb'))
dataframe = pd.read_csv('final_df.csv')
x = dataframe['text']
y = dataframe['real/fake']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    input_data = [news]
    vectorized_input_data = vectorization.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__,static_url_path='', static_folder="static", template_folder="templates")

app.config['DEBUG'] = True

with open("model_english_pipeline.pkl","rb") as handle:
	model = pickle.load(handle)
    
# A function for image to text conversion
class GetText(object):
    
    def __init__(self, file):
        self.file = pytesseract.image_to_string(Image.open(file))
    
# App routing is used to map the specific URL with the associated function that is intended to perform some task. 
@app.route('/')
def main():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
# A function to predict news from url specified using pipeline model
def predict():
    url = request.get_data(as_text = True)[5:]
    url = urllib.parse.unquote(url)
    try:
        article = Article(str(url))
        article.download()
        article.parse()
        article.nlp()
        news = article.summary
        #lang = TextBlob(news)
        #language = lang.detect_language()
        #print(lang.detect_language())
        language=langid.classify(news)
        language=language[0]
        if language == "hi" or language == "kn":
            translator = Translator()
            out = translator.translate(news, dest="en")
            #print(out.text)
            pred1 = model.predict([out.text])
            print(pred1)
            return render_template('index.html', prediction1 = pred1)
        else:
            pred1 = model.predict([news])
            print(pred1)
            return render_template('index.html', prediction1 = pred1)
    except:
        pred1=['NotUrl']
        return render_template('index.html', prediction1=pred1)
     
@app.route('/predict_text', methods=['POST'])
# A function to predict the text based news implemented using passive aggressive classifier
def predict_text():
    if request.method == 'POST':
        message = request.form['news']
        text = message
        #print(text)
        #lang = TextBlob(text)
        #language = lang.detect_language()
        #print(lang.detect_language())
        language=langid.classify(text)
        language=language[0]
        if message.startswith("https://" or "http://" or "www."):
            pred2=['Url']
            return render_template('index.html', prediction2=pred2)
        elif language == "hi" or language == "kn":
            translator = Translator()
            out = translator.translate(message, dest="en")
            print(out.text)
            pred2 = fake_news_det(out.text)
            print(pred2)
            return render_template('index.html', prediction2=pred2)        
        else:
            pred2 = fake_news_det(message)
            print(pred2)
            return render_template('index.html', prediction2=pred2)
    else:
        return render_template('index.html', prediction2="Something went wrong")
    
    
@app.route('/predict_image', methods=['GET','POST'])
def predict_image():
    if request.method == 'POST' or 'GET':
        try:
            if 'photo' not in request.files:
                pred3=["No photo"]
                return render_template('index.html', prediction3 = pred3)
            photo = request.files['photo']
            textObject = GetText(photo)
            if textObject.file.startswith("https://" or "http://" or"www."):
                pred3 = model.predict([textObject.file])
                return render_template('index.html', prediction3 = pred3)
                
                        
            elif len(textObject.file) <= 10:
                pred3=["No text"]
                return render_template('index.html', prediction3 = pred3)
            
            else:
                pred3 = fake_news_det(textObject.file)
                print(pred3)
                return render_template('index.html', prediction3 = pred3)
        except:
            pred3= ["Wrong format"]
            return render_template('index.html', prediction3 = pred3) 
    else:          
        return render_template('index.html',prediction3="Something went wrong")

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(port= port, debug=True, use_reloader=False)