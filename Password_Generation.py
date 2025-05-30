#!/usr/bin/env python
# coding: utf-8

## Creating Passwords using CHatGPt
import pandas as pd
from zxcvbn import zxcvbn
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.corpus import words

# Reading RockYou data set and creating List of Passwords to feed ChatGPT.
with open('rockyou.txt', 'r') as file:
    wordlist = file.read()

text = []
for x in range(len(wordlist)-5):
    word = "generate password with " + wordlist[x]+' '+wordlist[x+1]+' '+wordlist[x+2]+' '+wordlist[x+3]+' '+wordlist[x+4]
    text.append(word)

client = OpenAI(api_key='Enter Key')

answers = []
for prompt in text:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates one strong password."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    password = response.choices[0].message
    password = str(password)
    password = password.split("'")
    item = {'passwords':password[1]}
    answers.append(item)

nltk.download('words')
englishLIST = set(words.words())

# List of unwanted phrases to remove from GPT responses
removals = [
    'Note: Always ensure to use a password that is not related to easily available personal information and is a mix of alphanumeric and special characters',
    'Here are some password variations:',
    'Sure, here are a few password examples that are based on your inputs:',
    'Remember to make your password strong. It should have at least 12 characters, with a mix of numbers, upper and lower case letters, and symbols. Avoid any easily identifiable personal information, such as birthdays.',
    'Remember it is also crucial to make sure passwords are secure and do not contain any inappropriate or offensive words.',
    'Please note that it is never advised to use personal identifiable information like names or birthdays in your passwords, as it makes them more vulnerable to being cracked by hackers. Always try to use a random combination of letters, numbers & special characters.',
    'Please ensure to select a password which extends beyond personal and easily guessable parameters - this will ensure that your password is secure and cannot be easily hacked',
    'Make sure to change and secure your passwords regularly and avoid using personal information that others could easily know or guess',
    'This password includes all the names mentioned with alterations for characters for password strength. It also includes a special character and a number to enhance security.',
    'Please remember to save the password in a secure location, as it is important to keep your accounts and information safe.',
]

passwordGot = []
for entry in answers:
    temp = entry['passwords']
    # remove unwanted phrases
    for phrase in removals:
        temp = temp.replace(phrase, '')
    # remove numbering like "1.", "2.", etc.
    temp = re.sub(r"\b\d+\.\s*", '', temp)
    # remove non-ASCII characters (non-English keyboard keys)
    temp = temp.encode('ascii', errors='ignore').decode()
    # replace newlines with spaces
    temp = temp.replace('\n', ' ')
    # collapse multiple spaces
    temp = re.sub(r"\s+", ' ', temp).strip()
    passwordGot.append(temp)

# Extract the original prompts (without the leading instruction)
passwordsUsed = [t.replace('generate password with ', '') for t in text]

# Filter out English words from the generated strings
passwordGotList = []
for pwd in passwordGot:
    tokens = pwd.split(' ')
    filtered = [tok for tok in tokens if tok.lower() not in englishLIST]
    passwordGotList.append(filtered)

# Keep only tokens longer than 12 characters
passwordGotList2 = [[tok for tok in tokens if len(tok) > 12]
                     for tokens in passwordGotList]

# Pair each original prompt with its generated passwords
used, created = [], []
for orig, gen_list in zip(passwordsUsed, passwordGotList2):
    for new_pwd in gen_list:
        used.append(orig)
        created.append(new_pwd)

# Build DataFrame and save to CSV
df = pd.DataFrame({'original_prompt': used,
                   'generated_password': created})
# Save file passwords generated and the Prompts
df.to_csv('passwords.csv', index=False)

