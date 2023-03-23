#GeniusD\genius-expertise\genius-expertise\song_info.json
import json
import pandas as pd

#Edit these to match your paths 
SONG_INFO_PATH = "GeniusD\genius-expertise\genius-expertise\song_info.json"  #Refers to the song_info.json file
SONG_LYRICS_PATH =  "GeniusD\genius-expertise\genius-expertise\lyrics.jl"  #Refers to the lyrics_jl file


# Need to investigate these
#'News', 'Tracklist + Album Art'. 'Screen', 'Electronic Rock', 'Electro House', 'Jamaica', 
# 'Scandinavia', 'Electro', 'EDM', 'Horrorcore', 'Turkey', 'Tech', 'Electronica', 'Future Bass', 
#'Avant Garde', 'Neo-Psychedelia'
tag_stop_list = {'France', 'French Rap', 'Non-Music', 'Deutschland', 
                 'Deutscher Rap', 'Polska', 'Россия (Russia)', 'Polski Rap', 
                 'Русский рэп (Russian Rap)', 'En Español', 'Instrumental', 'Brasil',
                 'Em Português', 'News', 'Tracklist + Album Art', 'Latin Music', 
                 'Polska Muzyka', 'Korean', 'K-Pop (케이팝)', 'Österreich', 'Latin Trap', 'Dubstep', 
                 'Türkçe Sözlü Rap', 'Türkiye', 'Қазақстан (Kazakhstan)', 'Spanish Music'
                 , 'Puerto Rico','Русский трэп (Russian Trap)', 'Русское аренби (Russian R&B)',
                 'Русский рок (Russian Rock)', 'A Cappella', 'Русский поп (Russian Pop)', 'Japanese',
                 'Colombia', 'French R&B', 'Беларусь (Belarus)', 'Nederland', 
                 'België/Belgique', 'Italian Rap', 'Danmark'
                 }

#Make df 
df_song_lyrics = pd.read_json(SONG_LYRICS_PATH, lines = True) #37993
df_song_info = pd.read_json(SONG_INFO_PATH, lines= True)

#Join all the elements in song_lyrics where their song = url_name for the song info
#Need to do this because song info has a lot info about stuff not in this set 
df_merged = df_song_lyrics.merge(df_song_info, left_on = 'song', right_on='url_name') 
df_merged = df_merged[['primary_artist', 'title', 'tags', 'lyrics']]

#Get all the unique tags, we filter out tags with less than a 100 instances, and tags that indicate the song is not english
unique_tags = dict()
for tags in df_merged['tags']:
    for tag in tags:
        if tag in unique_tags:
            unique_tags[tag] = unique_tags[tag] + 1
        else:
            unique_tags[tag] = 1

tag_list = [(tag, unique_tags[tag]) for tag in unique_tags.keys()]
tag_list.sort(key = lambda x : x[1], reverse = True)
tag_list =  [tag_pair for tag_pair in tag_list if tag_pair[1] >= 100 and tag_pair[0] not in tag_stop_list]

df_merged.to_csv('ModData\\filter1.csv')







#Code to grab all unique tags across the entire dataset
# song_info = []
# unique_tags = dict()

# for line in open (SONG_INFO_PATH, 'r'):
#     song_info.append(json.loads(line))


# for info in song_info:
#     tags = info['tags']
#     for tag in tags:
#         if tag in unique_tags:
#             unique_tags[tag] = unique_tags[tag] + 1
#         else:
#             unique_tags[tag] = 1


# print(unique_tags)
