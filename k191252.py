import math
import os
from bs4 import BeautifulSoup
from matplotlib.pyplot import yscale
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os
import string
import nltk
from sklearn.naive_bayes import MultinomialNB
import spacy
from nltk.corpus import wordnet as wn
nlp=spacy.load('en_core_web_lg')
from nltk import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

remove_punctuation_translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))


N=821+230

def stopWord():
    global N
    stop_words=[]
    f=open("Stopword-List.txt")
    #make an array of all stop words
    for word in f.read().split("\n")[0:]:
        if word:
            stop_words.append(word.strip())
    return stop_words

#function used to read all files
def read_files():
    index={}
    
    stop_words=stopWord()
    df={}
    y=[]
    k=0
    #getcwd gets the file directory location
    direct=os.getcwd()
    direct=direct+"\\course-cotrain-data\\fulltext"
    for i in range(0,2):
        if i==0:
            files= os.listdir(direct+"\\course")
            tDirect=direct+"\\course"
            s="course"
        else:
            files= os.listdir(direct+"\\non-course")
            tDirect=direct+"\\non-course"
            s="non-course"
        for file in files:
            f=open(tDirect+"\\"+file)
            lines=f.read()

            #beautifulSoap is used to extract text from html file#
            soup = BeautifulSoup(lines, features="html.parser")

            for script in soup(["script", "style"]):
                script.extract()  
            text = soup.get_text()

            newWords=[]
            for items in text.split():
                #check if data contains any digit
                #if it does than discard the word
                #as it is not useful in classification
                if len(items)>1 and not any(ch.isdigit() for ch in items):
                    items=items.translate(remove_punctuation_translator)
                    newWords.append(lemmatizer.lemmatize(items.strip().lower()))
            
            #y saves the class of each document 
            y.append(s)
            temp=[]
            
            #making index
            for word in newWords:
                if word not in stop_words:
                    if word not in index:
                        index[word]=[0]*N
                    index[word][k]+=1
                    if word not in temp:
                        if df.__contains__(word):
                            df[word]+=1
                        else:
                            df[word]=1
                        temp.append(word)
        
            k=k+1
    return index,df,y


#function to select 50 features based on occurrence of nouns
def featureSelectionOnNouns(index,x_data):

    #pos_tag tell the pos of each token
    train_tokens=nltk.pos_tag(index.keys())

    tf={}
    for word in train_tokens:
        sum=0
        if word[1].startswith("N"):
            if word[0] not in tf:
                tf[word[0]]=0
            for i in range(0,N):
                sum+=index[word[0]][i]
            tf[word[0]]=sum

    #sorting total frequencies in descending order
    value_key_pairs = ((value,key) for (key,value) in tf.items())
    sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)

    #convert ordered dictionary into list
    tf = [list(item) for item in sorted_value_key_pairs]
    
    features=[]

    #append top 50 features into shared list X_DATA
    for count,word in enumerate(tf):
        if word not in features:
            x_data[word[1]]=[0]*1051
        features.append(index[word[1]])
        x_data[word[1]]=index[word[1]]
        if count+1==50:
            break
    return x_data,features



#function to select 100 features based on tf_idf
def featureSelectionOnTfIdf(index,df,features,x_data):
    #apply tf_idf on index
    normal_index,df=tfIdfScore(index,df)

    total_normalized_tf={}

    #sum tf_idf for each word
    for word in index:
        sum=0
        for i in range(0,N):
            sum+=normal_index[word][i]
        total_normalized_tf[word]=sum
    
    #sorting obtained total_tf
    value_key_pairs = ((value,key) for (key,value) in total_normalized_tf.items())
    sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)
    
    #converting ordered dictionary into list
    total_normalized_tf = [list(item) for item in sorted_value_key_pairs]
    total=100

    #append top 100 features into shared list X_DATA
    for count,word in enumerate(total_normalized_tf):
        if word[1] not in features:
            x_data[word[1]]=[0]*1051
        else:
            total=total+1
            continue
        features.append(word[1])
        x_data[word[1]]=index[word[1]]
        if count+1==total:
            break
        
    return x_data


#helper function used to normalize index 
#tf=1+log(tf[word])
#df=log(N/df[word])
def tfIdfScore(index,df):
    normal_index=index
    global N
    for word in normal_index:
        df[word]=math.log10(N/df[word])
        for i in range(0,N):
            normal_index[word][i]=(1+math.log10(normal_index[word][i]) if normal_index[word][i]>0 else 0)
            normal_index[word][i]=normal_index[word][i]*df[word]
    return normal_index,df



#fuction to get lexical chains for selected features to improve classification
def featureSelectionOnLexicalChain(x_data,index):
    nouns=[]
    #get all words from data
    for word in x_data.keys():
            nouns.append(word)
    for word in nouns:

        #find all syn for the words
        syn=[]
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                syn.append(lemma.name())
        #check similarity
        #if sim>.5 append into shared list X_DATA
        for i in range(len(syn)):
            if(nlp(syn[i]) and nlp(syn[i]).vector_norm):
                sim=nlp(word).similarity(nlp(syn[i]))
                if(sim>0.5):
                    x_data[syn[i]]=index[word]
    return x_data


def Normal_distribution(x, mean, std):
    prob_density = (1/(std*2.50))*np.exp(-0.5*((x-mean)/std)**2)
    return prob_density
    


#function for classification of text
#using multinomial naive bayes classifier
def Naive_Bayesian(X_train, X_test, y_train, y_test):
    for i in range(0):
        range_checker = 0
        signed_class = 0
        for j in X_train['Outcome'].unique():
            cal = 1
            prob_class = len(X_train[X_train['Outcome']==j]['Outcome'])/len(X_train['Outcome'])
            for features in X_train.columns:
                if features!='Outcome':
                    mean = np.mean(X_train[X_train['Outcome']==j][features])
                    std_div = np.std(X_train[X_train['Outcome']==j][features])
                    cal *= Normal_distribution(X_test.iloc[i][features], mean, std_div)
            answer = prob_class*cal
            if answer>range_checker:
                signed_class = j
                range_checker = answer
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    a=accuracy_score(y_test,predictions)
    p=precision_score(y_test, predictions, average = 'weighted')
    r=recall_score(y_test, predictions, average = 'weighted')

    f_measure=(2*p*r)/(p+r)
    return a,p,r,f_measure
        



x_data={}
index,df,y=read_files()
features=[]
print("Selecting Features")
print("Please wait for a few mintues....")
x_data,features=featureSelectionOnNouns(index,x_data)
x_data=featureSelectionOnTfIdf(index,df,features,x_data)
x_data=featureSelectionOnLexicalChain(x_data,index)
x=pd.DataFrame(x_data)
print("Running classifier.......")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
a,p,r,f_measure=Naive_Bayesian(X_train, X_test, y_train, y_test)

print("****************NAIVE BAYES CLASSIFIER RESULTS*********************")
print("Accuracy = ",a)
print("Precision = ",p)
print("Recall = ",r)
print("F-measure = ",f_measure)
