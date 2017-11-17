"""
File to compute headword of Quora questions
"""
# In[]:
# Import required libraries
import pickle

import numpy as np
import pandas as pd
import nltk
import spacy

from spacy import displacy


# In[]:
# Settings
DEMO = False
QUESTION_WORDS = ['if', 'can', 'what', 'when', 'where', 'why', 'which', 'who', 'how', 'whose', 'whom']


# In[]:
# Helper functions
def filter_headword(token_children):
    for child in token_children:
        # Child is class spacy.token
        # Need to convert to text

        # Find first noun in root verb's children not in question words
        if child.pos_ == 'NOUN' and not any(child.text.lower() in word for word in QUESTION_WORDS):
            return child.text


def get_headword(string):
    doc = nlp(string)
    for token in doc:
        # Find root verb and fetch to 2nd helper on top
        if token.dep_ == 'ROOT':
            return filter_headword(token.children)


# In[]:
# Load data

# Load Spacy model
nlp = spacy.load('en')

# Load dataset
df = pd.read_csv('../data/processed/train.csv')
if DEMO:
    df = df.head(500)

print(df.shape)

q1s = df['question1']
q2s = df['question2']

print('Loading done!')


# In[]:
# Headword examples
# To find question headword, go to ROOT under token.dep, and find first children
# That is not a wh-word
if not DEMO:

    # Spacy does dependency parsing and POS tagging automatically
    doc = nlp(u'Which city in China has the largest number of foreign financial companies?')
    for token in doc:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children])

    get_headword(u'Which city in China has the largest number of foreign financial companies?')

    doc2 = nlp('What is the state flower of California?')
    for token in doc2:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children])

    get_headword('What is the state flower of California?')

    doc3 = nlp('What should I do to be a great geologist?')
    for token in doc3:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children])

    get_headword('Is chocolate milk good for you?')

    get_headword('What is the painting on this image?')
    get_headword('What is this painting?')
    get_headword("What should I do if I forget my email password")
    get_headword('Can a bald person ever grow their hair back?')
    get_headword('Where does the new space come from?')


    df.head(50)

    get_headword(df['question1'][40])
    get_headword(df['question2'][40])

    df = df.iloc[0:50]
    sam1 = df.apply(lambda x: get_headword(x['question1']), axis=1)
    sam2 = df.apply(lambda x: get_headword(x['question2']), axis=1)
    df['headword_q1'] = sam1
    df['headword_q2'] = sam2

    get_headword('')


# In[]:
# Headword extraction

q1_headwords = []
q2_headwords = []

for i, (q1, q2) in enumerate(zip(q1s, q2s)):

    if i % 1000 == 0:
        print(i)

    # Multiple headwords for multiple sentences
    q1 = list(filter(None, q1.strip().replace('.', '?').split('?')))
    q2 = list(filter(None, q2.strip().replace('.', '?').split('?')))

    q1_headwords_sub = []
    for sentence in q1:
        q1_headwords_sub.append(get_headword(sentence))

    q2_headwords_sub = []
    for sentence in q2:
        q2_headwords_sub.append(get_headword(sentence))

    q1_headwords.append(q1_headwords_sub)
    q2_headwords.append(q2_headwords_sub)

print('Headword extraction done!')


# In[]:
# Pickle data

pickle.dump(q1_headwords, open('../data/features/q1_headwords.pkl', 'wb'))
pickle.dump(q2_headwords, open('../data/features/q2_headwords.pkl', 'wb'))

print('Pickle done!')
