import numpy as np
import pandas as pd
import sklearn
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import accuracy_score

def input_data(filename):
    df = pd.read_csv(filename)
    df.dropna(inplace = True)
    df = df[df['Rating'] != 3]
    return df


def stem(sentence):
    stemmer = SnowballStemmer("english")
    stemmed_sentence = ""
    for x in sentence.split(" "):
        stemmed_sentence += str(stemmer.stem(x)) + " "
    return stemmed_sentence

def test_train(df):
    df1 = df['Reviews']
    X = df1.values
    
    
    df['Rate'] = np.where(df['Rating']>3,1,0)
    df2 =  df['Rate']
    Y = df2.values
    return train_test_split(X,Y,random_state = 0)

def clean_data(X):
    refined_X = [] 
    for sen in X:
        sen = re.sub(r'[^a-zA-Z ]', ' ', sen)
        stop = stopwords.words('english')
        ref_sen=""
        for word in sen.split(" "):
            if word not in stop:
                ref_sen += word + " "
        ref_sen = re.sub(r'[ ]+',' ',ref_sen)
        ref_sen = stem(ref_sen)
        refined_X.append(ref_sen.strip())
    return refined_X

def bag_of_words(x):
    vec = CountVectorizer(min_df = 5,ngram_range=(1, 2)).fit(x)
    return vec

def tfidf(x):
    vec = TfidfVectorizer(min_df = 5,ngram_range=(1, 2)).fit(x)
    return vec
    
   
df = input_data("Amazon_Unlocked_Mobile.csv")
x_train, x_test, y_train, y_test = test_train(df)

def process(typ):
	
	x_train = clean_data(x_train)
	x_test = clean_data(x_test)

	with open("x_train.txt",'w') as fp:
		pickle.dump(x_train,fp)
		
	with open("x_test.txt",'w') as fp:
		pickle.dump(x_test,fp)

	return x_train,x_test, y_train, y_test


def load():
	with open("x_train.txt",'r') as fp:
		x_train = pickle.load(fp)
		
	with open("x_test.txt",'r') as fp:
		x_test = pickle.load(fp)
		
	return x_train,x_test, y_train, y_test


