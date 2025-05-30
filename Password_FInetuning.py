#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import os
from openai import OpenAI

# List of special characters
special_characters = [
    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
    '-', '_', '=', '+', '[', '{', ']', '}', '\\', '|',
    ';', ':', '\'', '"', ',', '<', '.', '>', '/', '?',
    '~', '`', ' ', '\t', '\n'
]


# Function to check if a string contains numbers
def contains_number(s):
    return any(char.isdigit() for char in s)

# Function to check if a string contains special characters
def contains_special_character(s):
    return any(char in special_characters for char in s)

def contains_number_and_special(s):
    return contains_number(s) and contains_special_character(s)

def replace_special_chars(s):
    for char in special_characters:
        if char in s:
            # Replace the character with a different one from the list
            new_char = random.choice([c for c in special_characters if c != char])
            s = s.replace(char, new_char)
    return s

df = pd.read_csv('/Users/afam/Downloads/Password Survey Paper/password list/training.csv')
df['password'] = df['Password'].apply(replace_special_chars)
df.head()

user_words = df[['Password1', 'Password2', 'Password3', 'Password4', 'Password5']].values.tolist()
passwords = df['password'].tolist()
train_list = []

for x in range(len(passwords)):
    word = user_words[x]
    train_list.append({
        "messages": [
            {"role": "system", "content": "You create users passwords given words."},
            {"role": "user", "content": f"Generate a secure password using {word[0]}, {word[1]}, {word[2]}, {word[3]}, {word[4]}:"},
            {"role": "assistant", "content": passwords[x]}
        ]
    })

with open('training.jsonl', 'w') as jsonl_file:
    for data in train_list:
        jsonl_file.write(json.dumps(data) + "\n")

print("JSONL file created successfully!")

# Initialize the OpenAI client
client = OpenAI(
    api_key="Enter Key"
)

#Upload the file
uploaded_file = client.files.create(
    file=open("training.jsonl", "rb"),
    purpose="fine-tune"
)

# Access the file ID using the attribute instead of subscript
file_id = uploaded_file.id
print(f"Uploaded File ID: {file_id}")

#fine-tuning job
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3
    }
)

# Print fine-tuning job ID
print(f"Fine-tuning Job ID: {fine_tune_job.id}")


# Using the FineTuned Job
completion = client.chat.completions.create(
  model="ft:gpt-4o-mini-2024-07-18:personal::AaRpiD6d",
  messages=[
    {"role": "system", "content": "You create users passwords given words."},
    {"role": "user", "content": f"Generate a secure password using Afamefuna, Promise, Umejiaku, Nelly, Iheruo:"}
  ]
)
print(completion.choices[0].message.content)
strength = zxcvbn(completion.choices[0].message.content)
print(f"Password Strength: {strength['score']} / 4, Guesses_log10 {strength['guesses_log10']}")

