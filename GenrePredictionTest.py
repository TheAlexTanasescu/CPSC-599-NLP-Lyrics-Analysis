from Cosine import getClosest
from collections import Counter
import os

# print(os.listdir())

PRE1 = "GeniusTest"
exists = 0

genreExists = Counter(["RNB", "Pop", "Country", "Rap"])

c = 0
for directory in os.listdir(PRE1) :
    # print(directory)
    for file_name in os.listdir(PRE1+"\\"+directory):   
        # print(file_name)
        if file_name == ".": continue
        with open(PRE1+"\\"+directory+"\\"+file_name, "r", encoding = 'cp850') as f:
            new_lines = f.readlines()
            new_content = "".join(new_lines)
            curr_preds = getClosest(new_content)
            print(file_name)
            c+=1
            for prediction in curr_preds:
                if "r&b" in prediction[0].lower() and directory == "RNB":
                    print("rnb", file_name)
                    print(curr_preds)
                    genreExists["RNB"] +=1
                elif "pop" in prediction[0].lower() and directory == "Pop":
                     genreExists["Pop"] +=1
                     print("pop", file_name)
                     print(curr_preds)
                elif "country" in prediction[0].lower() and directory == "Country":
                     genreExists["Country"] +=1
                     print("county", file_name)
                     print(curr_preds)
                elif "rap" in prediction[0].lower() and directory == "Rap":
                     genreExists["Rap"] +=1
                     print("rap", file_name)
                     print(curr_preds)

print(c)
print(genreExists)



