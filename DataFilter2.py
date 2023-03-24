import pandas as pd
import re
import string
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')


#FLAGS
CREATE_FILTER_2 = True
CREATE_FILTER_3 = True

#File Paths - These are written for a windows os- modify as required
DATA_FILTERED_PATH1 ='ModData\\filter1.csv' #After running DataFilter1.py
DATA_FILTERED_PATH2 = 'ModData\\filter2.csv' #Contains lyrics with regex remvoed
DATA_FILTERED_PATH3 = 'ModData\\filter3.csv' #Contains lyrics with tokenization + top stop words removal


stop_words = set(stopwords.words('english'))

#Regex to remove things inside of [] and the [] themselves
def remove_brackets(text):
    cleaned_text = re.sub(r'\[[^]]*\]|\([^)]*\)', '', text)
    return cleaned_text

def remove_break_chars(text):
    cleaned_text = re.sub(r'[\n\t]', ' ', text)
    return cleaned_text

def remove_regex(text):
    cleaned_text = remove_break_chars(remove_brackets(text))
    return cleaned_text

def remove_punctuation(text):
    # Remove punctuation from the text
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize(text):
    # Tokenize the text into individual words using nltk
    return nltk.word_tokenize(text)

def lemmatize(tokens):
    # Lemmatize the tokens using nltk
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token).lower() for token in tokens if len(token) > 2]

def remove_top_tokens(tokens, percent):
    # Count the frequency of each token
    token_counts = Counter(tokens)
    
    # Calculate the number of tokens to remove (top percent%)
    num_tokens_to_remove = int(len(token_counts) * (percent / 100))
    
    # Remove the top n tokens from the list of tokens
    top_tokens = [token for token, count in token_counts.most_common(num_tokens_to_remove)]
    return [token for token in tokens if token not in top_tokens and token not in stop_words]

def clean_and_tokenize_corpus(corpus):
    # Remove punctuation from the text
    corpus = remove_punctuation(corpus)
    
    # Tokenize the text into individual words using nltk
    tokens = tokenize(corpus)
    
    # Lemmatize the tokens using nltk
    tokens = lemmatize(tokens)
    
    # Remove the top 1% of tokens
    tokens = remove_top_tokens(tokens, 1)
    
    return tokens

def generate_filtered():

    if CREATE_FILTER_2:
        df = pd.read_csv(DATA_FILTERED_PATH1).iloc[:, 1:]
        df['lyrics'] = df['lyrics'].apply(remove_regex)
        df.to_csv(DATA_FILTERED_PATH2)

    #This implies that the file filter2.csv exists
    if CREATE_FILTER_3:
        df = pd.read_csv(DATA_FILTERED_PATH2).iloc[:,1:]
        df['lyrics'] = df['lyrics'].apply(clean_and_tokenize_corpus)
        df.to_csv(DATA_FILTERED_PATH3)

generate_filtered()


