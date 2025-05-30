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
import pandas as pd
from nltk.corpus import words
import nltk


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

# Open the file in read mode
with open('/Users/afam/Documents/Password Cleaned.txt', 'r') as file:
    # Read the lines of the file into a list
    lines = file.readlines()
    x = 0
text = []

for line in lines[1:]:
    temp = line.split(',')
    temp2 = [word for word in temp[2].split(' ') if word]
    text1 = []
    for y in temp2:
        y = str(y).replace('\n', '')
        text2 = []
        text2.append(y)
        text2.append(temp[1])
        text1.append(text2)
    text += text1
    x +=1
x = 0
List = []

for line in lines[1:]:
    temp = line.split(',')
    temp2 = [word for word in temp[2].split(' ') if word]
    List1 = []
    for y in temp2:
        y = str(y).replace('\n', '')
        List2 = []
        temp1 = temp[1].split(' ')
        List2.append(y)
        List2+=temp1
        List1.append(List2)
    List += List1
    x +=1
df = pd.DataFrame(List, columns=['passwordGot', 'Password1','Password2','Password3','Password4','Password5'])
df = df[df['passwordGot'].apply(len) >= 8] # Remove password of length less that 8
df

# List of special characters
special_characters = [
    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
    '-', '_', '=', '+', '[', '{', ']', '}', '\\', '|',
    ';', ':', '\'', '"', ',', '<', '.', '>', '/', '?',
    '~', '`', ' ', '\t', '\n'
]
# List of 10 special characters
special_10_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']

def trim_element(element, minimum, maximum):
    # Check if the element is a string and its length is greater than maximum
    if isinstance(element, str) and len(element) > maximum:
        # Look for a special character between positions minimum and maximum
        for i in range(minimum, maximum+1):
            if element[i] in special_characters:  # Add other special characters if needed
                return element[:i]
        # If no special character is found, truncate to maximum characters
        return element[:maximum]
    return element

def guesses(password,A,B,C,D,E):
    results = zxcvbn(password, user_inputs=[A,B,C,D,E])
    return(results['guesses_log10'])

# Function to check if a string contains numbers
def contains_number(s):
    return any(char.isdigit() for char in s)

# Function to check if a string contains special characters
def contains_special_character(s):
    return any(char in special_characters for char in s)

def contains_number_and_special(s):
    return contains_number(s) and contains_special_character(s)

# Function to replace special characters
def replace_special_chars(s):
    for char in special_characters:
        if char in s:
            # Replace the character with a different one from the list
            new_char = random.choice([c for c in special_characters if c != char])
            s = s.replace(char, new_char)
    return s

df['len8'] = df['passwordGot'].apply(lambda x: trim_element(x, 8, 8)) # Passwords of length 8
df['len16'] = df['passwordGot'].apply(lambda x: trim_element(x, 16, 16)) # Passwords of length 16
df['range8_12'] = df['passwordGot'].apply(lambda x: trim_element(x, 8, 12)) #Passwords of minimum 8 and max 12
df['range13_17'] = df['passwordGot'].apply(lambda x: trim_element(x, 13, 17))#Passwords of minimum 13 and max 17
df

df['guessLen8'] = df.apply(lambda row: guesses(row['len8'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df['guessLen16'] = df.apply(lambda row: guesses(row['len16'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df['guessRange8_12'] = df.apply(lambda row: guesses(row['range8_12'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df['guessRange13_17'] = df.apply(lambda row: guesses(row['range13_17'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)

# Standard passwords of Length 8
df_len8 = df[df['len8'].apply(contains_number_and_special)]
#y = df_len8['guessLen8'].apply(len)==8
orig_8 = df_len8['guessLen8']
sns.violinplot(orig_8)
plt.axhline(6, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 8 Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
#plt.xlabel('X-axis')
plt.savefig('plot_8.png') 
plt.show()

# Standard passwords of Length 16
df_len16 = df[df['len16'].apply(contains_number_and_special)]
df_len16 = df_len16[df_len16['len16'].apply(len)==16] #making sure passwords are lenght 16
orig_16 = df_len16['guessLen16']
sns.violinplot(orig_16)
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 16 Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^{12}$','$10^{14}$','$10^{16}$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.savefig('plot_16.png') 
plt.show()

# Standard passwords of range8_12
df_range8_12 = df[df['range8_12'].apply(contains_number_and_special)]
orig_8_12 = df_range8_12['guessRange8_12']
sns.violinplot(orig_8_12)
plt.axhline(6, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for Range 8-12 Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^{12}$','$10^{14}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()

# Standard passwords of range13_17
df_range13_17 = df[df['range13_17'].apply(contains_number_and_special)]
orig_13_17 = df_range13_17['guessRange13_17']
sns.violinplot(orig_13_17)
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for range 13-17 Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^{12}$','$10^{14}$','$10^{16}$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()

pd.concat([orig_8, orig_8_12, orig_16, orig_13_17], axis=1).describe()

df_len8['len8_modified'] = df['len8'].apply(replace_special_chars)
df_len16['len16_modified'] = df['len16'].apply(replace_special_chars)
df_range8_12['range8_12_modified'] = df['range8_12'].apply(replace_special_chars)
df_range13_17['range13_17_modified'] = df['range13_17'].apply(replace_special_chars)


df_len8['guessLen8_modified'] = df_len8.apply(lambda row: guesses(row['len8_modified'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df_len16['guessLen16_modified'] = df_len16.apply(lambda row: guesses(row['len16_modified'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df_range8_12['guessRange8_12_modified'] = df_range8_12.apply(lambda row: guesses(row['range8_12_modified'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df_range13_17['guessRange13_17_modified'] = df_range13_17.apply(lambda row: guesses(row['range13_17_modified'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)


# Standard passwords of Length 8
sns.violinplot(df_len8['guessLen8_modified'])
plt.axhline(6, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 8 Modified Special Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()

# Standard passwords of Length 16
sns.violinplot(df_len16['guessLen16_modified'])
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 16 Modified-Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^{12}$','$10^{14}$','$10^{16}$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()


# Standard passwords of range8_12
sns.violinplot(df_range8_12['guessRange8_12_modified'])
plt.axhline(6, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for Range 8-12 Modified Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^{12}$','$10^{14}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()

# Standard passwords of range13_17
sns.violinplot(df_range13_17['guessRange13_17_modified'])
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for Range 13-17 Modified Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^{12}$','$10^{14}$','$10^{16}$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()

sym_8 = df_len8['guessLen8_modified']
sym_16 = df_len16['guessLen16_modified']
sym_8_12 = df_range8_12['guessRange8_12_modified']
sym_13_17 = df_range13_17['guessRange13_17_modified']

def swap_special_with_numbers(series):
    def swap_positions(s):
        s_list = list(s)
        i = 0
        while i < len(s_list):
            if s_list[i] in special_characters:
                for j in range(i+1, len(s_list)):
                    if s_list[j].isdigit():
                        # Swap positions
                        s_list[i], s_list[j] = s_list[j], s_list[i]
                        break
            i += 1
        return ''.join(s_list)

    return series.apply(swap_positions)

#swapped_series = swap_special_with_numbers(data)

df_len8['len8_swap'] = swap_special_with_numbers(df_len8['len8_modified'])
df_len16['len16_swap'] = swap_special_with_numbers(df_len16['len16_modified'])
df_range8_12['range8_12_swap'] = swap_special_with_numbers(df_range8_12['range8_12_modified'])
df_range13_17['range13_17_swap'] = swap_special_with_numbers(df_range13_17['range13_17_modified'])


df_len8['guessLen8_swaped'] = df_len8.apply(lambda row: guesses(row['len8_swap'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df_len16['guessLen16_swaped'] = df_len16.apply(lambda row: guesses(row['len16_swap'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df_range8_12['guessRange8_12_swaped'] = df_range8_12.apply(lambda row: guesses(row['range8_12_swap'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)
df_range13_17['guessRange13_17_swaped'] = df_range13_17.apply(lambda row: guesses(row['range13_17_swap'], row['Password1'],row['Password2'], row['Password3'],row['Password4'], row['Password5']), axis=1)

swap_8 = df_len8['guessLen8_swaped'] 
swap_16 = df_len16['guessLen16_swaped']
swap_8_12 = df_range8_12['guessRange8_12_swaped']
swap_13_17 = df_range13_17['guessRange13_17_swaped'] 

# Standard passwords of Length 8
sns.violinplot(df_len8['guessLen8_swaped'])
#sns.violinplot(df_len8['guessLen8'], color = 'r')   #original 8
plt.axhline(6, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 8 swaped Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.savefig('8.png', dpi=300)
plt.show()
df_len8['guessLen8_swaped'].describe()

# Standard passwords of Length 16
sns.violinplot(df_len16['guessLen16_swaped'])
#sns.violinplot(df_len8['guessLen8'], color = 'r')   #original 8
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 16 Swapped Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^12$','$10^14$','$10^16$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.savefig('16.png', dpi=300)
plt.show()
df_len16['guessLen16_swaped'].describe()

# Standard passwords of Length 8-12
sns.violinplot(df_range8_12['guessRange8_12_swaped'] )
plt.axhline(6, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 8-12 Swapped Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^12$','$10^14$','$10^16$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.savefig('8-12.png', dpi=300)
plt.show()
df_range8_12['guessRange8_12_swaped'] .describe()

# Standard passwords of Length 13-17
sns.violinplot(df_range13_17['guessRange13_17_swaped'])
#sns.violinplot(df_len8['guessLen8'], color = 'r')   #original 8
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 13-17 Swapped Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^12$','$10^14$','$10^16$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.savefig('13-17.png', dpi=300)
plt.show()
df_range13_17['guessRange13_17_swaped'].describe()

# Standard passwords of Length 13-17
sns.violinplot(df_range13_17['guessRange13_17_swaped'])
#sns.violinplot(df_len8['guessLen8'], color = 'r')   #original 8
plt.axhline(10, color = 'r', linestyle = '--', label = 'Recommended Online Guessing Limit')
plt.title('Guessing Time for 16 Swapped Character Passwords')
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^12$','$10^14$','$10^16$','$10^{18}$']# Define custom tick labels 
plt.yticks(ticks=tick_positions, labels=tick_labels)# Set custom y-ticks
plt.show()
df_range13_17['guessRange13_17_swaped'].describe()

final_df = pd.concat([orig_8, orig_8_12, orig_16, orig_13_17, sym_8, sym_8_12, sym_16, sym_13_17, swap_8, swap_8_12, swap_16, swap_13_17], axis=1)
final_df.columns = ['LLM_Password_Len8', 'LLM_Password_Len8-12', 'LLM_Password_Len16', 'LLM_Password_Len13-17',
                    'Password_symbol_changed_Len8', 'Password_symbol_changed_Len8-12', 'Password_symbol_changed_Len16', 'Password_symbol_changed_Len13-17',
                    'Password_position_swap_Len8', 'Password_position_swap_Len8-12', 'Password_position_swap_Len16', 'Password_position_swap_Len13-17']
final_df.to_csv('Final_data.csv', index=False)

final_df.describe().T

# Initialize the matplotlib figure with subplots
fig, axes = plt.subplots(4, 3, figsize=(20, 10))  # 2 rows, 3 columns

# First row: Length 8
tick_positions = [0, 2, 4, 6, 8, 10]  # Custom tick positions
tick_labels = ['$10^0$', '$10^2$', '$10^4$', '$10^6$', '$10^8$', '$10^{10}$']  # Custom tick labels

sns.violinplot(y=orig_8, ax=axes[0, 0])
axes[0, 0].set_title('LLM Generated Password Len8')
axes[0, 0].axhline(6, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[0, 0].set_yticks(tick_positions)
axes[0, 0].set_yticklabels(tick_labels)
axes[0, 0].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=sym_8, ax=axes[0, 1])
axes[0, 1].set_title('Password Symbol Changed Len8')
axes[0, 1].axhline(6, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[0, 1].set_yticks(tick_positions)
axes[0, 1].set_yticklabels(tick_labels)
axes[0, 1].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=swap_8, ax=axes[0, 2])
axes[0, 2].set_title('Password Symbol Changed & Position Swap Len8')
axes[0, 2].axhline(6, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[0, 2].set_yticks(tick_positions)
axes[0, 2].set_yticklabels(tick_labels)
axes[0, 2].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

# Second row: Length 8-12
tick_positions = [0, 2, 4, 6, 8, 10, 12, 14]
tick_labels = ['$10^0$', '$10^2$', '$10^4$', '$10^6$', '$10^8$', '$10^{10}$', '$10^{12}$', '$10^{14}$']

sns.violinplot(y=orig_8_12, ax=axes[1, 0])
axes[1, 0].set_title('LLM Generated Password Len(8-12)')
axes[1, 0].axhline(6, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[1, 0].set_yticks(tick_positions)
axes[1, 0].set_yticklabels(tick_labels)
axes[1, 0].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=sym_8_12, ax=axes[1, 1])
axes[1, 1].set_title('Password Symbol Changed Len(8-12)')
axes[1, 1].axhline(6, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[1, 1].set_yticks(tick_positions)
axes[1, 1].set_yticklabels(tick_labels)
axes[1, 1].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=swap_8_12, ax=axes[1, 2])
axes[1, 2].set_title('Password Symbol Changed & Position Swap Len(8-12)')
axes[1, 2].axhline(6, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[1, 2].set_yticks(tick_positions)
axes[1, 2].set_yticklabels(tick_labels)
axes[1, 2].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label


# Third row: Length 16
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^12$','$10^14$','$10^16$','$10^{18}$']# Define custom tick labels 

sns.violinplot(y=orig_16, ax=axes[2, 0])
axes[2, 0].set_title('LLM Generated Password Len(16)')
axes[2, 0].axhline(10, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[2, 0].set_yticks(tick_positions)
axes[2, 0].set_yticklabels(tick_labels)
axes[2, 0].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=sym_16, ax=axes[2, 1])
axes[2, 1].set_title('Password Symbol Changed Len(16)')
axes[2, 1].axhline(10, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[2, 1].set_yticks(tick_positions)
axes[2, 1].set_yticklabels(tick_labels)
axes[2, 1].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=swap_16, ax=axes[2, 2])
axes[2, 2].set_title('Password Symbol Changed & Position Swap Len(16)')
axes[2, 2].axhline(10, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[2, 2].set_yticks(tick_positions)
axes[2, 2].set_yticklabels(tick_labels)
axes[2, 2].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

# Third row: Length 13-17
tick_positions = [0, 2, 4, 6, 8, 10,12,14,16,18]# Define custom tick positions
tick_labels = [ '$10^0$','$10^2$','$10^4$','$10^6$','$10^8$','$10^{10}$','$10^12$','$10^14$','$10^16$','$10^{18}$']# Define custom tick labels 

sns.violinplot(y=orig_13_17, ax=axes[3, 0])
axes[3, 0].set_title('LLM Generated Password Len(13-17)')
axes[3, 0].axhline(10, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[3, 0].set_yticks(tick_positions)
axes[3, 0].set_yticklabels(tick_labels)
axes[3, 0].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=sym_13_17, ax=axes[3, 1])
axes[3, 1].set_title('Password Symbol Changed Len(13-17)')
axes[3, 1].axhline(10, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[3, 1].set_yticks(tick_positions)
axes[3, 1].set_yticklabels(tick_labels)
axes[3, 1].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

sns.violinplot(y=swap_13_17, ax=axes[3, 2])
axes[3, 2].set_title('Password Symbol Changed & Position Swap Len(13-17)')
axes[3, 2].axhline(10, color='r', linestyle='--', label='Recommended Online Guessing Limit')
axes[3, 2].set_yticks(tick_positions)
axes[3, 2].set_yticklabels(tick_labels)
axes[3, 2].set_ylabel('Guess Attempts (log scale)')  # Add y-axis label

plt.savefig("Dictionary.png", dpi=300, bbox_inches='tight')
# Adjust layout for better spacing
plt.tight_layout()
plt.show()
