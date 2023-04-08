import re
import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ssl
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from collections import Counter

ssl._create_default_https_context = ssl._create_unverified_context


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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



# Function to clean the lyrics
def clean_lyrics(lyrics):
    # Remove non-alphanumeric characters and lowercase
    lyrics = re.sub(r'[^a-zA-Z0-9\s]', '', lyrics).lower()
    # Tokenize the lyrics
    tokens = nltk.word_tokenize(lyrics)
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    #cleaned_lyrics = ' '.join(tokens)
    return tokens

# Load the lyrics
# with open('baby.rtf', 'r') as f:
#     lyrics = f.read()



 
S_L = ""
while (1):
    nl = str(input())
    if nl == "`" : break
    S_L = S_L + nl

lyrics = S_L  
print(S_L)

# Clean the lyrics
lyrics = clean_and_tokenize_corpus(lyrics)

#print(sentences)
# Load the pre-trained Doc2Vec model
model = Doc2Vec.load('artists_model')

'''
print("Lyrics: ")
print(lyrics)
print("length of lyrics: ")
print( str(len(lyrics)))
'''

new_embedding = model.infer_vector(lyrics)

#print("embeddings")
#print(embeddings)


# Load other embeddings from a .npy file
genre_embeddings = np.load('genre_embeddings.npy',allow_pickle=True).item()

#genre_embeddings_array = np.array(genre_embeddings_array).reshape(1,-1)


#print("genre_embedding")
#print(genre_embeddings_array)

# Find the closest embeddings using cosine similarity
similarities = []
#similarities = cosine_similarity(embeddings, genre_embeddings_array)
#similarities = cosine_similarity(embeddings, genre_embeddings_array[0])
# similarities = [cosine_similarity(embeddings, np.array(genre_embeddings_array[i]).reshape(1,-1)) for i in range(len(genre_embeddings_array))]
sim1 = []

for genre, embedding in genre_embeddings.items():
    cos_sim = cosine(new_embedding, embedding)
    # cos_sim1 = cosine_similarity([new_embedding], [embedding])[0][0]
    similarities.append((genre,  cos_sim))
    # sim1.append((genre, cos_sim1))

# print(similarities)

#Sort by highest
similarities.sort(key = lambda x: x[1], reverse=False)
# sim1.sort(key = lambda x: x[1], reverse= True)


num_closest = 10

# Print the closest embeddings
for i in range (num_closest):
        print(similarities[i])
