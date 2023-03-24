import pandas as pd
import os 


#FLAGS 
#SET TO TRUE IF YOU WANT TO GENERATE ON UR MACHINE
GENERATE_GENRE = False
GENERATE_ARTIST = False

#File Paths
DATA_FILTERED_PATH3 = 'ModData\\filter3.csv' #Contains lyrics with tokenization + top stop words removal
GENRE_DATA = "LyricDataByArtist"
path_suffix = GENRE_DATA + "\\"     #MODIFY FOR MAC OS
ARTIST_DATA = "ArtistsData"
path_suffix1 = ARTIST_DATA + "\\" 

df = pd.read_csv(DATA_FILTERED_PATH3).iloc[:,1:]

unique_artists = dict()
#grab the unique artists
for artist in  df['primary_artist']:

    if artist in unique_artists:
        unique_artists[artist] = unique_artists[artist] + 1

    else:
        unique_artists[artist] = 1

# unique_artists_list = [(artist, unique_artists[artist]) for artist in unique_artists.keys()]
# unique_artists_list.sort(key = lambda x: x[1], reverse=True)
unique_artists_list = [artist for artist in unique_artists.keys()]

unique_tags = set()

def str_to_list(s):
    return eval(s)



df['tags'] = df['tags'].apply(str_to_list)


for tags in df['tags']:
    for tag in tags:
        unique_tags.add(tag)



#MODIFY PATHS FOR MAC OS
unique_tags = list(unique_tags)

if GENERATE_GENRE:
    for tag in unique_tags:
        dir_path_tag = path_suffix + "\\" +  tag
        if tag not in os.listdir(GENRE_DATA):
            os.makedirs(dir_path_tag)
        bool_mask = df['tags'].apply(lambda x: tag in x)
        tag_df = df[bool_mask] #All the artists with a specfic tag

        for artist in tag_df['primary_artist']:
            file_path = dir_path_tag + "\\" + artist + ".csv"
            artist_tag_df = tag_df[tag_df['primary_artist'] == artist]
            artist_tag_df.to_csv(file_path)

if GENERATE_ARTIST:
    for artist in unique_artists_list:
        dir_path_tag = path_suffix1
        if ARTIST_DATA not in os.listdir():
            os.makedirs(dir_path_tag)
        bool_mask = df['primary_artist'] == artist
        artist_df = df[bool_mask] #Get a specific artist
        file_path = dir_path_tag + "\\" + artist +".csv"
        artist_df.to_csv(file_path)

    




        

    




        



    