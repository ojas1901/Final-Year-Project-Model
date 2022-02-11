import numpy as np
import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
def clean_text(text):
    text = text.lower()
    #newline
    text=text.replace("\n"," ").replace("\r"," ")
    #punc and numbers
    punc_list='!@+-"#$%^&*)(,./:;<>?[\]_{|}~' +'0123456789'
    t=str.maketrans(dict.fromkeys(punc_list," "))
    text=text.translate(t)
    #single quote 
    t=str.maketrans(dict.fromkeys("'`",""))
    text=text.translate(t)
    #extra white space
    text=text.strip()
    text = ' '.join(text.split())
    return text   

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inpu = request.get_json()
    inp=inpu['question']
   # print(inp)
    #inp=request.get_data()
    #inp=inp.decode()
    #print(inp)
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(data)
    inp = clean_text(inp)
    #print(inp)
    inp_series = pd.Series(inp)
    inp_vector = loaded_vectorizer.transform(inp_series)
    prediction = model.predict(inp_vector.toarray())
    final_prediction=""
    for predictions in prediction:
        final_prediction=predictions

   # print(type(final_prediction))
    #print(final_prediction)
    return final_prediction
    #return render_template('index.html', prediction_text=prediction)
    

if __name__ == "__main__":
    port = int(os.getenv('PORT'))
    app.run(debug=True)
