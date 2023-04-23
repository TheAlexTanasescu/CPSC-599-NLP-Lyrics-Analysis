
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
import pyLDAvis._display
import pyLDAvis.gensim_models as gensimvis
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud




nltk.download('wordnet')
#Preprocessing Datasets\data\SongLyricsDataset\csv\ArianaGrande.csv


#Flags  

LEMMA_FLAG = True
STOP_TOP = True
STOP_TOP_PERCENT = 0.25
SAVE_TO_DISK = True
SHOW_WORD_CLOUD_LDA = False

stopWords = set(stopwords.words('english')).union({'wanna', 'gotta', 'like', 'got', 'yeah', 'let', 'gonna', 'yet', 'released', 'wan', 'gon', 'song', 'lyrics', 'released', 'unreleased'})
lemmatizer = WordNetLemmatizer()

artistLyrics = {}
artistFreq = {}
artistStopWords = {}

#For TF-IDF you see better results with ignoring the top stop word removal thing 

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

                if (LEMMA_FLAG == True):
                    tokens = [lemmatizer.lemmatize(token).lower() for token in tokens]
                tokens = [token.lower() for token in tokens if token not in stopWords]
                tokens = [token for token in tokens if (token.isalpha() and len(token)>2)]

                artistLyrics[artistName].append(lyric)
                

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
#do LDA and TFIDF for these artists
lda_models = []
corpora = []
id2words = []
tf_idf_res = []
for artist in artists: 
    texts = artists[artist]
    artLyrics = artistLyrics[artist]
    # print(type(texts[0]))
    id2word = Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    corpora.append(corpus)
    id2words.append(id2word)
    #LDA
    # lda_model = LdaModel(corpus=corpus, id2word=id2word,num_topics=10, random_state=100,update_every=1,chunksize=100,passes=1,alpha='auto',per_word_topics=True)
    lda_model = LdaModel(corpus=corpus, id2word=id2word,num_topics=10, random_state=100,update_every=1,passes=1,alpha='auto',per_word_topics=True)
    lda_models.append((artist, lda_model)) 
                      
    #TF-idf
    vectorizer = TfidfVectorizer( stop_words=list(stopWords.union(artistStopWords[artist])))
    X = vectorizer.fit_transform(artLyrics)
    tf_idf_res.append((artist, X, vectorizer))
    featureNames = vectorizer.get_feature_names_out()
    sorted_tfidf_matrix = np.argsort(X.toarray())[:, ::-1]
    top_feature_names = [[ featureNames[j] for j in i[ : k]] for i,k in zip(sorted_tfidf_matrix, [5,5,5,5]) ]

    print("TFIDF:", top_feature_names)
        #print(artist, lda_model.print_topics())
    with open("ldaResults.txt", "a") as f:
        print(artist)
        if (artist == 'bts (방탄소년단)'): continue
        f.write(artist)
        top10 = lda_model.show_topics(num_topics=10, formatted = True)
        for top in top10:
            print(top)
            f.write(str(top)+"\n")


load_from = []

if(SAVE_TO_DISK):
    os.chdir('ldaModels')
    for i, model in enumerate(lda_models):
        model[1].save(model[0]+'lda')
        load_from.append(model[0]+'lda')

# print(load_from)



#Visualizations--------------------

#Visualizing TF-IDF----
prefix ="tfidfVis\\"
for i, res in enumerate(tf_idf_res):
    vectorizer = res[2]
    tfidf_matrix = res[1]
    artist_name = res[0]
    # Create a DataFrame from the TF-IDF matrix
    terms = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.todense(), columns=terms)

    if not os.path.exists(prefix+artist_name):
        os.makedirs(prefix+artist_name)
        # print(f"The directory '{prefix+artist_name}' has been created.")

    #Heatmap -----------
    top_terms = df_tfidf.mean().sort_values(ascending=False)[:10].index.tolist()
    df_top_terms = df_tfidf[top_terms]
    # Create a heatmap of the TF-IDF scores for the top terms
    sns.set(font_scale=0.5)
    hm = sns.heatmap(df_top_terms, cmap='Blues', cbar=False)
    plt.title('TF-IDF Scores for Top 10 Terms')
    plt.xlabel('Terms')
    plt.ylabel('Documents')
    # plt.show()
    # fig = hm.get_figure()
    # fig.savefig(prefix+artist_name+"\\"+artist_name+"hm.png")

    #WordCloud -----------
    # Create a WordCloud from the TF-IDF matrix
    wc = WordCloud(width = 1200, height = 800, background_color='white', max_words=50)
    wc.generate_from_frequencies(df_tfidf.sum(axis=0))
    # Display the WordCloud
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    # plt.show()
    wc.to_file(prefix+artist_name+"\\"+artist_name+"wc.png")

    



# #Visualizing LDA -----
# #Word Cloud: 
# for i, model in enumerate(lda_models):
#     if SHOW_WORD_CLOUD_LDA == False: break
#     topics = model[1].show_topics(num_topics=10, formatted = False)
#     for j, topic in topics:
#         plt.figure()
#         plt.imshow(WordCloud(background_color='white').fit_words(dict(topic)))
#         plt.axis('off')
#         plt.title(f'Artist: {model[0]}, Topic {j+1}')
#         plt.show()


os.chdir("..")
#Using pyLDAvis
#Use interactive mode/notebook form to iterate through visualizations to see what each one looks like
visualizations = []
prefix = "ldaVis\\"
for i, model in enumerate(lda_models):
    artist_name = model[0]
    # pyLDAvis.enable_notebook()
    vis = gensimvis.prepare(model[1], corpora[i], id2words[i])
    visualizations.append(vis)
    pyLDAvis.save_html(vis, prefix+artist_name+'.html')




