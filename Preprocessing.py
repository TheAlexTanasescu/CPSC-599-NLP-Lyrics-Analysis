
#File Formats and Locations:

# Song Lyrics from 79 musical genres https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?resource=download
#has a seperate dataset for artists
# Datasets\data\songlyrfrom79\artists-data.csv
# Format: Artist Genre Songs Popularity Link

# 5 Million Song Lyrics Dataset https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset
# Datasets\data\bigSong\ds2.csv
#Format:  SongTitle GenreTag Artist Year Views FeaturedArtist Lyrics ID

#Song Lyrics Datset
# Datasets\data\SongLyricsDataset\csv
#Format: Artist,Title,Album,Date,Lyric,Year

import re
import os 
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('wordnet')
#Preprocessing Datasets\data\SongLyricsDataset\csv\ArianaGrande.csv


#Flags  

LEMMA_FLAG = True
STOP_TOP = True
STOP_TOP_PERCENT = 0.025

stopWords = set(stopwords.words('english')).union({'wanna', 'gotta'})
lemmatizer = WordNetLemmatizer()

artistLyrics = {}
artistFreq = {}
artistStopWords = {}

def preProcessSongLyricsDataset( filename = "Datasets\data\SongLyricsDataset\csv"):

    result= {}

    for artist in os.listdir(filename):

        first = True 
        newContent = pd.read_csv(filename+"\\"+artist)
        artistName = newContent['Artist'][0].lower()
        # print(artistName)

        if first:
            result[artistName] = []
            artistLyrics[artistName]= []
            artistStopWords[artistName] = set()
            first = False
        
        wordFreq = FreqDist()
         
        for lyric in newContent['Lyric']:
            # print(type(lyric))
            # Dataset has the issue that some songs dont have lyrics attatched to them. 
            try:
                
                tokens = word_tokenize(lyric)

                if (STOP_TOP == True):
                    for token in tokens:
                        wordFreq[token]+=1

                tokens = [token.lower() for token in tokens if token not in stopWords]
                tokens = [token for token in tokens if token.isalnum()]
                artistLyrics[artistName].append(lyric)
                
                if (LEMMA_FLAG == True):
                    tokens = [lemmatizer.lemmatize(token).lower() for token in tokens]

            except:
                # print(artistName)
                continue
            # result[artistName].append(lyric)

            result[artistName].append(tokens)

        if STOP_TOP == True:
            artistFreq[artistName] = wordFreq
            uniqueWords = len(wordFreq)
            artistStop = int(STOP_TOP_PERCENT * uniqueWords)
            topStopWords = set([pair[0] for pair in wordFreq.most_common(artistStop)])
            artistStopWords[artistName] = artistStopWords[artistName].union(topStopWords)

            #Filter them out 
            for lyric in result[artistName]:
                tokens = [token for token in lyric if token not in artistStopWords]


    # print(result)
    return result

# print(os.listdir())
artists = preProcessSongLyricsDataset()



#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

for artist in artists: 
    texts = artists[artist]
    artLyrics = artistLyrics[artist]
    print(type(texts[0]))
    id2word = Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    #LDA
    # lda_model = LdaModel(corpus=corpus, id2word=id2word,num_topics=10, random_state=100,update_every=1,chunksize=100,passes=1,alpha='auto',per_word_topics=True)
    lda_model = LdaModel(corpus=corpus, id2word=id2word,num_topics=10, random_state=100,update_every=1,passes=1,alpha='auto',per_word_topics=True)
    #TF-idf
    vectorizer = TfidfVectorizer( stop_words=list(stopWords.union(artistStopWords[artist])))
    X = vectorizer.fit_transform(artLyrics)
    featureNames = vectorizer.get_feature_names_out()
    sorted_tfidf_matrix = np.argsort(X.toarray())[:, ::-1]
    top_feature_names = [ [featureNames[j] for j in i[:k]] for i,k in zip(sorted_tfidf_matrix, [5,5,5,5]) ]
    
    print("TFIDF:", top_feature_names)
        #print(artist, lda_model.print_topics())
    print(artist, lda_model.show_topics(num_words =10, formatted = False))
   #Tfidf 


