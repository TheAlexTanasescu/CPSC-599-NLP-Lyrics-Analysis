import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Set up NLTK and download required resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define a function to preprocess the lyrics text
def preprocess_text(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    # Return the filtered tokens as a space-separated string
    return ' '.join(filtered_tokens)

# Define a function to generate Doc2Vec embeddings for a given set of lyrics
def generate_embedding(model, lyrics):
    preprocessed_lyrics = preprocess_text(lyrics)
    tagged_lyrics = TaggedDocument(preprocessed_lyrics.split(), [0])
    embedding = model.infer_vector(tagged_lyrics.words)
    return embedding

# Set up parameters for Doc2Vec model training
vector_size = 100
window_size = 5
min_count = 1
num_workers = 4
num_epochs = 100

# Set up parameters for t-SNE projection
perplexity = 5
n_iter = 1000

main_folder = 'LyricDataByArtist/'
# Iterate over the subfolders (genres) in the main folder
for genre_folder in os.listdir(main_folder):
    genre_path = os.path.join(main_folder, genre_folder)
    
    # Iterate over the CSV files (artists) in the subfolder (genre)
    for artist_file in os.listdir(genre_path):
        artist_path = os.path.join(genre_path, artist_file)
        
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(artist_path)
        
        # Preprocess the lyrics column and store as a list
        preprocessed_lyrics = [preprocess_text(lyrics) for lyrics in df['lyrics']]
        
        # Train a Doc2Vec model for the artist
        tagged_lyrics = [TaggedDocument(lyrics.split(), [i]) for i, lyrics in enumerate(preprocessed_lyrics)]
        model = Doc2Vec(tagged_lyrics, vector_size=vector_size, window=window_size, min_count=min_count, workers=num_workers, epochs=num_epochs)
        
        # Generate an embedding for each song in the CSV file
        embeddings = [generate_embedding(model, lyrics) for lyrics in preprocessed_lyrics]
        
        # Store the embeddings in a new column in the DataFrame
        df['embedding'] = embeddings
        
        # Project the embeddings using t-SNE
        if len(df) >= perplexity + 1:
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
            X_tsne = tsne.fit_transform(np.array(df['embedding'].values.tolist()))

            # Plot the projected embeddings
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
            plt.title(f'{genre_folder} - {artist_file}')
            plt.show()
        else:
            print(f"Skipping {genre_folder}/{artist_file}: not enough data points for perplexity {perplexity}")
