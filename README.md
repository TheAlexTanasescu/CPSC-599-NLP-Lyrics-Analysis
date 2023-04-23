# CPSC-599-NLP-Lyrics-Analysis

TO GENERATE THE DATASET ON YOUR LOCAL MACHINE:
1. Run DataFilter1.py
2. Run DataFilter2.py
3. Run genArtLyrics.py

This will generate two primary datasets:
a. ArtistData
b. LyricDataByArtist

In ArtistData you will find csv filess which have the format: primary_artist, title, tags, lyrics. And contain all the songs from a specific artist
In LyricDatabyArtist, you will find directories for each artist with music in that genre, and find csv files containing the lyrics for all their music with that genre tag

NOTE: You may have to modify these file paths in the code in accordance with your operating system's file sys. This was written using a windows os. 

Preprocessing contains the results from lda+tf-idf
Cosine deals with artist similarity 
