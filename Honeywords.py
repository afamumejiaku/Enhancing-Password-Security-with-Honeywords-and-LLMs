#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from openai import OpenAI
# Initialize the OpenAI client
client = OpenAI(
    api_key="Enter Key"
)

# Step 1: Read the CSV file
df = pd.read_csv('/home/afam/Downloads/training.csv')
df.head(2)

Aux = []
start_all = time.time()
for idx, row in df.iterrows():
    prompt = (
        "Generate 20 secure passwords using the following words: "
        f"{row['Password1']}, {row['Password2']}, "
        f"{row['Password3']}, {row['Password4']}, {row['Password5']}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You generate secure passwords from given words."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
            n=20
        )
    except Exception as e:
        print(f"[Row {idx}] API error: {e}")
        continue

    # Clean up the model output
    raw = resp.choices[0].message.content
    lines = [line.strip() for line in raw.splitlines()]
    pwds = [
        re.sub(r'^\d+\.\s*', '', line)
        for line in lines
        if line
    ]
    Aux.append(pwds)

Honeywords = []
password_index = []
for index, row in df.iterrows():
    start_time = time.time()
    Honey = []
    for x in range(19):
        completion = client.chat.completions.create(
          model="ft:gpt-4o-mini-2024-07-18:personal::AcEJyk6k",
          messages=[
            {"role": "system", "content": "You create users passwords given words."},
            {"role": "user", "content": f"Generate a secure password using {row['Password1']},{row['Password2']},{row['Password3']},{row['Password4']},{row['Password5']}:"}
          ]
        )
        Honey.append(completion.choices[0].message.content)
    pwd_index = random.randint(0, 19)
    Honey.insert(pwd_index,row['Password']) 
    Honeywords.append(Honey)
    password_index.append(pwd_index)

Honeywords2 = []
password_index2 = []
for index, row in df.iterrows():
    Honey2 = []
    for x in range(19):
        completion = client.chat.completions.create(
          model="ft:gpt-4o-mini-2024-07-18:personal::AaRpiD6d",
          messages=[
            {"role": "system", "content": "You create users passwords given words."},
            {"role": "user", "content": f"Generate a secure password using {row['Password1']},{row['Password2']},{row['Password3']},{row['Password4']},{row['Password5']}:"}
          ]
        )
        Honey2.append(completion.choices[0].message.content)
    pwd_index2 = random.randint(0, 19)
    Honey2.insert(pwd_index2,row['Password']) 
    Honeywords2.append(Honey2)
    password_index2.append(pwd_index2)

