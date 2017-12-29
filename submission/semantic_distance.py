'''
Semantic distance scoring functions, as described by Mihalcea et all (2006)
Features described in the `feature_engineering/wordnet_distance.ipynb` notebook

'''

import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy
# Load spaCy pipeline. Disable Named Entity Recognition, since I don't need it. 
nlp = spacy.load('en_core_web_lg', disable=['ner'])


##############################
# Sentence semantic similarity scoring functions
##############################

# POS tagging
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None

# get the most common synset for a tagged word from WordNet. 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        # get the first synset of the word in WordNet
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def path_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    
    # Zip up the synsets and the words/POS tags
    zipped1 = zip(sentence1,synsets1)
    zipped2 = zip(sentence2,synsets2)
    
    # filter out the Nones
    zipped1 = [z for z in zipped1 if z[1] is not None]
    zipped2 = [z for z in zipped2 if z[1] is not None]
 
 
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
 
    # For each word in the first sentence
    for tup1 in zipped1:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tup1[1].path_similarity(ss[1]) for ss in zipped2 if \
                              penn_to_wn(ss[0][1]) == penn_to_wn(tup1[0][1])])
        except:
            best_score = None
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for tup2 in zipped2:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tup2[1].path_similarity(ss[1]) for ss in zipped1 if \
                              penn_to_wn(ss[0][1]) == penn_to_wn(tup2[0][1])])
        except:
            best_score = None
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    try:
        # Average the values and add score from both sides to get symmetic distance
        score = .5*(score1/count1 + score2/count2)
        return(score)
    except:
        return(None)


def lch_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    
    # Zip up the synsets and the words/POS tags
    zipped1 = zip(sentence1,synsets1)
    zipped2 = zip(sentence2,synsets2)
    
    # filter out the Nones
    zipped1 = [z for z in zipped1 if z[1] is not None]
    zipped2 = [z for z in zipped2 if z[1] is not None]
    
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
    
    # For each word in the first sentence
    for tup1 in zipped1:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tup1[1].lch_similarity(ss[1]) for ss in zipped2 if \
                              penn_to_wn(ss[0][1]) == penn_to_wn(tup1[0][1])])
        except:
            best_score = None
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for tup2 in zipped2:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tup2[1].lch_similarity(ss[1]) for ss in zipped1 if \
                              penn_to_wn(ss[0][1]) == penn_to_wn(tup2[0][1])])
        except:
            best_score = None
            
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    try:
        # Average the values and add score from both sides to get symmetic distance
        score = .5*(score1/count1 + score2/count2)
        return(score)
    except:
        return(None)


def wup_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    
    # Zip up the synsets and the words/POS tags
    zipped1 = zip(sentence1,synsets1)
    zipped2 = zip(sentence2,synsets2)
    
    # filter out the Nones
    zipped1 = [z for z in zipped1 if z[1] is not None]
    zipped2 = [z for z in zipped2 if z[1] is not None]
 
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
 
    # For each word in the first sentence
    for tup1 in zipped1:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tup1[1].wup_similarity(ss[1]) for ss in zipped2 if \
                              penn_to_wn(ss[0][1]) == penn_to_wn(tup1[0][1])])
        except:
            best_score = None
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for tup2 in zipped2:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tup2[1].wup_similarity(ss[1]) for ss in zipped1 if \
                              penn_to_wn(ss[0][1]) == penn_to_wn(tup2[0][1])])
        except:
            best_score = None
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    # Average the values and add score from both sides to get symmetic distance
    try:
        score = .5*(score1/count1 + score2/count2)
        return(score)
    except:
        return(None)

def embedding_similarity_score(sentence1, sentence2):
    """ compute the sentence similarity using Pre-trained Word2Vec embeddings from spaCy. """
    
    # Process text - extract POS and embeddings
    doc1 = nlp(unicode(sentence1))
    doc2 = nlp(unicode(sentence2))
    
    # Get a list of tokens, only for those tokens which are not stopwords or punctuation
    tokens1 = [token for token in doc1 if token.text not in stops and token.pos_ != u'PUNCT']
    tokens2 = [token for token in doc2 if token.text not in stops and token.pos_ != u'PUNCT']
    
    # accumulate the Cosine similiarities between vectors, and number of matched vectors. 
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
 
    # For each word in the first sentence
    for tok1 in tokens1:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tok1.similarity(tok2) for tok2 in tokens2])
        except Exception as e:
            best_score = None
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for tok2 in tokens2:
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([tok2.similarity(tok1) for tok1 in tokens1])
        except Exception as e:
            best_score = None
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    try:
        # Average the values and add score from both sides to get symmetic distance
        score = .5*(score1/count1 + score2/count2)
        return(score)
    except:
        return(None)





##############################
# Extract the semantic similarity scores used in the classifier
##############################

def extract_sematnic_similarity(df):
    
    # Leacock Chodorow Distace
    df['lch_similarity'] = df.apply(lambda x : lch_similarity(x['question1'],x['question2']), axis = 1)
    # Pairwise GloVe embedding cosine distance
    df['embedding_similarity_score'] = df.apply(lambda x : embedding_similarity_score(x['question1'],x['question2']), axis = 1)

    return(df)






















