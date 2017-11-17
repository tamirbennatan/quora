"""
File to get named entities off Quora questions
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
# Load data
DEMO = False

# Load Spacy model
nlp = spacy.load('en', disable=['parser'])

# Load dataset
df = pd.read_csv('../data/processed/train.csv')
if DEMO:
    df = df.head(500)

print(df.shape)

q1s = df['question1']
q2s = df['question2']

print('Loading done!')


# In[]:
# Named Entity Recognition
# Using Spacy's built-in neural net model

q1_ents_text = []
q2_ents_text = []
q1_ents = []
q2_ents = []

for i, (q1, q2) in enumerate(zip(q1s, q2s)):

    if i % 1000 == 0:
        print(i)

    doc1 = nlp(q1)
    doc2 = nlp(q2)

    # Every question has a list of ents
    q1_ents_text_sub = []
    q1_ents_sub = []
    for ent in doc1.ents:
        q1_ents_text_sub.append(ent.text)
        q1_ents_sub.append((ent.text, ent.label, ent.label_))

    q2_ents_text_sub = []
    q2_ents_sub = []
    for ent in doc2.ents:
        q2_ents_text_sub.append(ent.text)
        q2_ents_sub.append((ent.text, ent.label, ent.label_))

    q1_ents_text.append(q1_ents_text_sub)
    q1_ents.append(q1_ents_sub)

    q2_ents_text.append(q2_ents_text_sub)
    q2_ents.append(q2_ents_sub)

print('NER done!')


# In[]:
# Pickle data

pickle.dump(q1_ents_text, open('../data/features/q1_ents_text.pkl', 'wb'))
pickle.dump(q1_ents, open('../data/features/q1_ents.pkl', 'wb'))

pickle.dump(q2_ents_text, open('../data/features/q2_ents_text.pkl', 'wb'))
pickle.dump(q2_ents, open('../data/features/q2_ents.pkl', 'wb'))

print('Pickle done!')


# In[]:
# Example
text = """But Google is starting from behind. The company made a late push
into hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa
software, which runs on its Echo and Dot devices, have clear leads in
consumer adoption."""

doc = nlp(text)
displacy.render(doc, style='ent', jupyter=True)

a = pickle.load(open('../data/features/q1_ents_text.pkl', 'rb'))
b = pickle.load(open('../data/features/q2_ents_text.pkl', 'rb'))
a
