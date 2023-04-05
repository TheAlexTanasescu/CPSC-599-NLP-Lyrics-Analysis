import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

import numpy as np
import seaborn as sns

def visGenreEmbeddings():
    # Load the genre embeddings dictionary from disk
    genre_embeddings = np.load('genre_embeddings.npy', allow_pickle=True).item()

    # Get the genre names and embeddings as separate arrays
    genre_names = list(genre_embeddings.keys())
    genre_vectors = np.array(list(genre_embeddings.values()))

    # Compute t-SNE embeddings
    tsne_embeddings = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(genre_vectors)


    # Plot the 2D embeddings with labels and colors
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='both', labelsize=12)

    # Use a colormap to assign a color to each point
    cmap = plt.get_cmap('gist_rainbow')
    norm = plt.Normalize(vmin=0, vmax=len(genre_names)-1)

    for i, genre in enumerate(genre_names):
        x, y = tsne_embeddings[i]
        color = cmap(norm(i))
        ax.scatter(x, y, color=color, label=genre)
        ax.annotate(genre, (x, y),fontsize=10)
    # ax.legend()
    plt.show()



def visArtistEmbeddings(size = None):
    # Load the artist embeddings dictionary from disk
    artist_embeddings = np.load('artist_embeddings.npy', allow_pickle=True).item()

    # Get the artist names and embeddings as separate arrays
    artist_names = list(artist_embeddings.keys())
    artist_vectors = np.array(list(artist_embeddings.values()))

    # Compute t-SNE embeddings
    tsne_embeddings = TSNE(n_components=2, perplexity=35, learning_rate=200).fit_transform(artist_vectors)


    if size == None:
        # Plot the 2D embeddings with labels and colors
        fig, ax = plt.subplots(figsize=(100, 100))
        ax.tick_params(axis='both', labelsize=12)

        # Use a colormap to assign a color to each point
        cmap = plt.get_cmap('gist_rainbow')
        norm = plt.Normalize(vmin=0, vmax=len(artist_names)-1)

        for i, artist in enumerate(artist_names):
            x, y = tsne_embeddings[i]
            color = cmap(norm(i))
            ax.scatter(x, y, color=color, label=artist)
            ax.annotate(artist, (x, y),fontsize=1)
        # ax.legend()
        plt.show()




visArtistEmbeddings()