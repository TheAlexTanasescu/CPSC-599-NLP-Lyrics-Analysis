import re
import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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
with open('baby.rtf', 'r') as f:
    lyrics = f.read()

# Clean the lyrics
lyrics = clean_lyrics(lyrics)

#print(sentences)
# Load the pre-trained Doc2Vec model
model = Doc2Vec.load('artists_model')

'''
print("Lyrics: ")
print(lyrics)
print("length of lyrics: ")
print( str(len(lyrics)))
'''

embeddings = model.infer_vector(lyrics)
embeddings = np.array(embeddings).reshape(1,-1)

#print("embeddings")
#print(embeddings)


# Load other embeddings from a .npy file
other_embeddings = np.load('genre_embeddings.npy',allow_pickle=True).item()
other_embeddings_array = [value for value in other_embeddings.values()]
#other_embeddings_array = np.array(other_embeddings_array).reshape(1,-1)


#print("other_embedding")
#print(other_embeddings_array)

# Find the closest embeddings using cosine similarity
#similarities = cosine_similarity(embeddings, other_embeddings_array)
#similarities = cosine_similarity(embeddings, other_embeddings_array[0])
similarities = [cosine_similarity(embeddings, np.array(other_embeddings_array[i]).reshape(1,-1)) for i in range(len(other_embeddings_array))]
print(similarities)
closest_indexes = np.argmax(similarities, axis=1)

# Print the closest embeddings
for index in closest_indexes:
    if index in other_embeddings:
        print(other_embeddings[index])
