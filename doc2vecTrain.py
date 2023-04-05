import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import pandas as pd

# Set up parameters for Doc2Vec model
vector_size = 100
window_size = 5
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
epochs = 50
workers = 6
dm = 0  # 0 for PV-DBOW, 1 for PV-DM

## Create an empty list to store the lyrics from all csvs
all_lyrics = []

# Iterate through all csvs inside the ArtistsData directory
for filename in os.listdir('ArtistsData'):
    if filename.endswith('.csv'):
        # Load the csv as a pandas dataframe
        df = pd.read_csv(os.path.join('ArtistsData', filename))
        # Add the lyrics column to the list of all lyrics
        all_lyrics += df['lyrics'].tolist()

# Create a list of TaggedDocuments, where each document is a lyric and has a unique tag
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_lyrics)]

# Train a Doc2Vec model on the list of TaggedDocuments
model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=50)

# Save the trained model to disk
model.save('artists_model')