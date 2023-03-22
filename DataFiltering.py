#GeniusD\genius-expertise\genius-expertise\song_info.json
import json

SONG_INFO_PATH = "GeniusD\genius-expertise\genius-expertise\song_info.json"

song_info = []
unique_tags = dict()

for line in open (SONG_INFO_PATH, 'r'):
    song_info.append(json.loads(line))


for info in song_info:
    tags = info['tags']
    for tag in tags:
        if tag in unique_tags:
            unique_tags[tag] = unique_tags[tag] + 1
        else:
            unique_tags[tag] = 1


print(unique_tags)

tag_list = [(tag, unique_tags[tag]) for tag in unique_tags.keys()]
tag_list.sort(key = lambda x : x[1], reverse = True)