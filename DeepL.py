# import libraries
import deepl
import pandas as pd

api_key = "DeepL_API" # should be replaced with a valid API key
translator = deepl.Translator(api_key)


# read the dataset
df = pd.read_csv('rlc_test.csv')
original_src = df["text_orig"]
correct_src = df["text_cor"]


# translate original sentences and save them in a .txt file

original_target = []

for line in original_src:
    translation = translator.translate_text(line, source_lang="RU", 
                                            target_lang="EN-US")
    original_target.append(translation.text)

with open("original_target_deepl.txt", "w") as f:
    for s in original_target:
        f.write(s + "\n")


# translate corrected sentences and save them in a .txt file

correct_target = []

for line in correct_src:
    translation = translator.translate_text(line, source_lang="RU", 
                                            target_lang="EN-US")
    correct_target.append(translation.text)

with open("correct_target_deepl.txt", "w") as f:
    for s in correct_target:
        f.write(s + "\n")