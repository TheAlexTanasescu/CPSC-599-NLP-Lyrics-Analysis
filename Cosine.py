import re
import nltk
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import ssl
from scipy.spatial.distance import cosine

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
# with open('baby.rtf', 'r') as f:
#     lyrics = f.read()



 
S_L = ""
while (1):
    nl = str(input())
    if nl == "`" : break
    S_L = S_L + nl

lyrics = S_L    

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
    similarities.append((genre, cos_sim))
    # sim1.append((genre, cos_sim1))

# print(similarities)

#Sort by highest
similarities.sort(key = lambda x: x[1], reverse=False)
# sim1.sort(key = lambda x: x[1], reverse= False)


num_closest = 5

# Print the closest embeddings
for i in range (num_closest):
        print(similarities[i])
