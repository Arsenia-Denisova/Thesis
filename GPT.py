# import libraries
import openai
import pandas as pd
from openai import OpenAI

openai.api_key = 'OpenAI_API' # should be replaced with a valid API key


# read the dataset
df = pd.read_csv('rlc_test.csv')
original_src = df["text_orig"]
correct_src = df["text_cor"]


# translate original sentences and save them in a .txt file

original_target = []

for line in original_src:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
               "role": "system",
                "content": """You will be provided with a sentence in Russian, 
                and your task is to translate it into English."""
            },
            {
                "role": "user",
                "content": line
            }
        ],
        temperature=0.7,
        max_tokens=200,
        top_p=1
    )

    translation = response.choices[0].message.content
    original_target.append(translation)

with open("original_target_gpt.txt", "w") as f:
    for s in original_target:
        f.write(s + "\n")


# translate corrected sentences and save them in a .txt file

correct_target = []

for line in correct_src:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
               "role": "system",
                "content": """You will be provided with a sentence in Russian, 
                and your task is to translate it into English.""" 
            },
            {
                "role": "user",
                "content": line
            }
        ],
        temperature=0.7,
        max_tokens=200,
        top_p=1
    )

    translation = response.choices[0].message.content
    correct_target.append(translation)

with open("correct_target_gpt.txt", "w") as f:
    for s in correct_target:
        f.write(s + "\n")