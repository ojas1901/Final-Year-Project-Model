import numpy as np
import os
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import regexp_tokenize
from nltk.corpus import wordnet
from sklearn.ensemble import VotingClassifier

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

#https://github.com/barrust/pyspellchecker
def spell_check(text):
    spell = SpellChecker()
    word_list=[]
    for word in text:
        word_list.append(spell.correction(word))
    return word_list

#https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5
def tokenizer(text):
    words=regexp_tokenize(text,pattern='\s+',gaps=True)
    return words
def lemma_pos(stopwords_removed):
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None

    lemmatizer = WordNetLemmatizer()

    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(stopwords_removed) 
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged)) #original pos tagging

    #lemmatization + pos tagging list
    lemma_pos_dict = {}
    wordlist = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemma_pos_dict[word]=tag
            wordlist.append(word)
        else:       
            # else use the tag to lemmatize the token
            lemma = lemmatizer.lemmatize(word, tag)
            lemma_pos_dict[lemma]=tag
            wordlist.append(lemma)
    return lemma_pos_dict,wordlist    
def term_freq(data_words,data_words_verbs):
    weights={'v':5,'n':3,'a':3,'g':1,'r':1,None:1}
    dict={}
    summation=0
    for word in data_words:
        summation+=weights[data_words_verbs[word]]
    for word in data_words:
        if word not in dict.keys():
             dict[word]=(data_words.count(word) * weights[data_words_verbs[word]])/summation;
    return dict 
def idf(tf):
    import math
    idf={}
    for doc in tf:
        for word in doc.keys():
            count=0
            for docno in tf:
                if word in docno.keys() and word not in idf:
                    count+=1
          
            if(word not in idf):
                idf[word]=1+math.log(len(tf)/count,10)
            
    return idf

def tf_idf(l,idf_list):
    l1=[]

    for doc in l:
        tfidf={}
        for word in doc.keys():
            tfidf[word]=idf_list[word]*doc[word];
        l1.append(tfidf)
    return l1
def CSR_input(unique_words,tf_idf_dict):    
    import numpy as np
    import pandas as pd
    len_tfidf=len(tf_idf_dict)
    len_unique_words=len(unique_words)
    df = pd.DataFrame(np.zeros((len_tfidf, len(unique_words))))
    df.columns = unique_words
    sentno=0
    for sentence in tf_idf_dict:
        for word in sentence.keys():
            df.iloc[sentno][word]=sentence[word]
        sentno=sentno+1    
    from scipy import sparse
    sparse_tfidf=sparse.csr_matrix(df)
    #print(unique_words[:10])
    return sparse_tfidf        

app = Flask(__name__)
loaded_model = pickle.load(open('model.pkl', 'rb'))
#unique_words = pickle.load(open('vectorizer.pkl', 'rb'))

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
    inplist=[]
    inplist.append(inp)
    pp_sentence_listi=[]
    position_dict_listi=[]
    for sentencei in inplist:
        cleanedi = clean_text(sentencei)
        tokenizedi = tokenizer(cleanedi)
        spellcheckedi= spell_check(tokenizedi)
        #stopwords_removed = remove_stopwords(spellchecked)
        pos_dicti,wordlisti = lemma_pos(spellcheckedi)
        pp_sentence_listi.append(wordlisti)
        position_dict_listi.append(pos_dicti)
    tfi=[]
    for (sentencei, onedicti) in zip(pp_sentence_listi,position_dict_listi):
        outputi = term_freq(sentencei,onedicti)
        tfi.append(outputi)
    idf_listi=idf(tfi)
    tf_idf_dicti=tf_idf(tfi,idf_listi)
    with open('unique_words.pkl', 'rb') as f:
        unique_words_input = pickle.load(f)
    sparse_matrixi = CSR_input(unique_words_input, tf_idf_dicti)
    #print("sparse_matrixi")
    print(sparse_matrixi)
    sparse_inpi= sparse_matrixi
    predictedi = loaded_model.predict(sparse_inpi.toarray())
    print(predictedi)
    lis=str(predictedi.flat[0].tolist())
    
  
    
    return lis
    # print(type(final_prediction))
        #print(final_prediction)
    
    #return render_template('index.html', prediction_text=prediction)
    

if __name__ == "__main__":
    port = int(os.getenv('PORT'))
    app.run(debug=True)
