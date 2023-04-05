import os
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import numpy as np


model = Doc2Vec.load('artists_model')



dir_name = "ArtistEmbeddings\\"
dir_name1 = "GenreEmbeddings\\"

def generateArtistEmbeddings():
    # Create an empty dictionary to store the artist embeddings
    artist_embeddings = {}
    # Iterate through all csvs inside the ArtistsData directory
    for filename in os.listdir('ArtistsData'):
        if filename.endswith('.csv'):
            # Load the csv as a pandas dataframe
            df = pd.read_csv(os.path.join('ArtistsData', filename))
            # Extract the artist name from the filename
            artist_name = filename.split('.')[0]
            # Create an empty list to store the embeddings for this artist's lyrics
            artist_lyric_embeddings = []
            # Iterate through the lyrics in this dataframe and generate an embedding for each one
            for lyric in df['lyrics']:
                # Generate a Doc2Vec embedding for this lyric
                lyric = eval(lyric)
                embedding = model.infer_vector(lyric)
                # Append the embedding to the list for this artist
                artist_lyric_embeddings.append(embedding)
            # Average the embeddings for this artist's lyrics to generate a general embedding for the artist
            artist_embedding = np.mean(artist_lyric_embeddings, axis=0)
            # Store the artist embedding in the dictionary
            artist_embeddings[artist_name] = artist_embedding

            # Save the artist embedding to disk
            np.save(dir_name+artist_name + '_embedding.npy', artist_embedding)

    # Save the artist embeddings dictionary to disk
    np.save('artist_embeddings.npy', artist_embeddings)



def generateGenreEmbeddings():
    # Create an empty dictionary to store the genre embeddings
    genre_embeddings = {}

    # Iterate through all subdirectories inside the LyricDataByArtist directory
    for genre_dir in os.listdir('LyricDataByArtist'):
        if os.path.isdir(os.path.join('LyricDataByArtist', genre_dir)):
            # Create an empty list to store the embeddings for this genre's lyrics
            genre_lyric_embeddings = []
            # Iterate through all csvs inside this genre directory
            for filename in os.listdir(os.path.join('LyricDataByArtist', genre_dir)):
                if filename.endswith('.csv'):
                    # Load the csv as a pandas dataframe
                    df = pd.read_csv(os.path.join('LyricDataByArtist', genre_dir, filename))
                    # Create an empty list to store the embeddings for this song's lyrics
                    song_lyric_embeddings = []
                    # Iterate through the lyrics in this dataframe and generate an embedding for each one
                    for lyric in df['lyrics']:
                        # Generate a Doc2Vec embedding for this lyric
                        embedding = model.infer_vector(lyric.split())
                        # Append the embedding to the list for this song
                        song_lyric_embeddings.append(embedding)
                    # Average the embeddings for this song's lyrics to generate a general embedding for the song
                    song_embedding = np.mean(song_lyric_embeddings, axis=0)
                    # Append the song embedding to the list for this genre
                    genre_lyric_embeddings.append(song_embedding)
            # Average the embeddings for all the songs in this genre to generate a general embedding for the genre
            genre_embedding = np.mean(genre_lyric_embeddings, axis=0)
            # Store the genre embedding in the dictionary
            genre_embeddings[genre_dir] = genre_embedding

            # Save the genre embedding to disk
            np.save(dir_name1 + genre_dir + '_embedding.npy', genre_embedding)

    # Save the genre embeddings dictionary to disk
    np.save('genre_embeddings.npy', genre_embeddings)

generateGenreEmbeddings()