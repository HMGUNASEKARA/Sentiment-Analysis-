import pandas as pd 
import numpy as np 
import re    
import string
import nltk
import pickle

# Load the model
with open('static/model/model.pickle','rb') as f:
    model = pickle.load(f)

# Load stop wards
with open ('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# Load the tokernizwer
vocab = pd.read_csv('static/model/vocabulary.txt', header = None)
tokens = vocab[0].tolist() #  Convart the vocabulary in to regular python list

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def remove_punctuations(text):
    for punctuation in string.punctuation:   # THis mean for loop go one by one in string. punctuation and check  is ther are same 
        text = text.replace(punctuation, ' ')   # If find that kind of equation then remove that value and replace with  nuthing 
    return text 

def preprocessing(text):
    data = pd.DataFrame([text],columns = ['tweet'])
    # If input text have upper case we have to convart it in to the lower case
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Remove the links if there is an links 
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    # Remove punctuations 
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    # Remove the numbers 
    data["tweet"] = data['tweet'].str.replace(r'\d+', '', regex=True)
    # Remove the stop wards 
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    # Apply the stemming 
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorization(ds):  # enter the data set and vobulary 
    vectorized_lst = []   # create empty set after fill this and got as output 

    for sentence in ds:     
        sentence_lst = np.zeros(len(tokens))  # By going one by one of and our sentence and make the array have zeros
        for i in range (len(tokens)):   # Then we check the sentence and vocabiulary,
            if tokens[i] in sentence.split():  # If sentence have that ward put 1 otherwise put zero
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)  # After one sentence append that into the vectorized lst 
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return(vectorized_lst_new)


def get_prediction (text):
    prediction = model.predict(text)
    if prediction ==1:
        return 'Negative'
    else:
        return 'Positive'
    
