#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

import os
filename = "model.pkl"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'rb') as f:
    vect, tfidf_transformer, clf = pickle.load(f)
# with open('vect.pkl', 'rb') as f:
#     vect = pickle.load(f)
# with open('tfidf_transformer.pkl', 'rb') as f:
#     tfidf_transformer = pickle.load(f)
# with open('clf.pkl', 'rb') as f:
#     clf = pickle.load(f)
# model = pickle.load(open('vect.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])



def predict():
    
    def clean_text(text):

        def get_wordnet_pos(pos_tag):
            if pos_tag.startswith('J'):
                return wordnet.ADJ
            elif pos_tag.startswith('V'):
                return wordnet.VERB
            elif pos_tag.startswith('N'):
                return wordnet.NOUN
            elif pos_tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN


        # lower text
        text = text.lower()
        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        # join all
        text = " ".join(text)
        return(text)

    if request.method == 'POST':
        message = request.form['message']
        clean_data = clean_text(message)
        data_vect = vect.transform([clean_data])
        data_tf =  tfidf_transformer.fit_transform(data_vect)
        prediction = clf.predict(data_tf)
        if prediction == 0:
            html = 'social anxiety.html'
        elif prediction == 1:
            html = 'schizophrenia.html'
        elif prediction == 2:
            html = 'ptsd.html'
        elif prediction == 3:
            html = 'ocd.html'
        elif prediction == 4:
            html = 'depression.html'
        elif prediction == 5:
            html = 'bipolar.html'
        elif prediction == 6:
            html = 'dysthymia.html'
    return render_template(html)



if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




