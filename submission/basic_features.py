'''
Basic features to be extracted.
Features described in 'feature_engineering/baseline_features.ipynb file'
'''

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from operator import itemgetter
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords


"""
Functions for extracting basic features. 
These are copied from the notebook `feature_engineering/basic_features.ipynb`,
but put in this script for convenience - so that I can apply them to a new dataset through this script. 
"""

# Stopwords and a list of punctuation - which I will reference in this file. 
punctuation = ["?", "!", ",", ".", '"', "-"]
# nltk stopword list
stopwords = list(stopwords.words('english'))



####################################
# Syntactic features: including number of words, characters, etc in each question
####################################

# Number of words the two questions share
def len_intersection(df, stopwords = None):
    if stopwords:
        df['len_intersection_sw'] = df.apply(lambda x: len(set([w for w in x["question1"].strip().split(" ") if
                                    w not in punctuation and w not in stopwords]).intersection(
                    set([w for w in x["question2"].strip().lower().split(" ") if
                         w not in punctuation and w not in stopwords]))),
                    axis = 1)
        return(df)

    else:
        df['len_intersection'] = df.apply(lambda x: len(set([w for w in x["question1"].strip().split(" ") if
                                    w not in punctuation]).intersection(
                    set([w for w in x["question2"].strip().lower().split(" ") if
                         w not in punctuation]))),
                     axis = 1)
        return(df)

# Number of words in the first question
def num_words_q1(df, stopwords = None):
    if stopwords:
        df['num_words_q1_sw'] = df['question1'].apply(lambda x: len([w for w in x.strip().lower().split(" ")
                                                          if w not in punctuation and w not in stopwords]))
        return(df)
    
    else:
        df['num_words_q1'] = df['question1'].apply(lambda x: len([w for w in x.strip().lower().split(" ") if
                                                          w not in punctuation]))
        return(df)

# Number of words in the second question
def num_words_q2(df, stopwords = None):
    if stopwords:
        df['num_words_q2_sw'] = df['question2'].apply(lambda x: len([w for w in x.strip().lower().split(" ")
                                                          if w not in punctuation and w not in stopwords]))
        return(df)
    
    else:
        df['num_words_q2'] = df['question2'].apply(lambda x: len([w for w in x.strip().lower().split(" ")
                                                          if w not in punctuation]))
        return(df)

# difference in the number of words in question one and question two. 
# This function assumes that `num_words_q1` and `num_words_q2` were already called. 
def num_words_diff(df, stopwords = None):
    if stopwords:
        df['num_words_diff_sw'] = abs(df['num_words_q1_sw'] - df['num_words_q2_sw'])
        return(df)
    else:
        df['num_words_diff'] = abs(df['num_words_q1'] - df['num_words_q2'])
        return(df)

# number of character in the first question
def num_chars_q1(df, stopwords = None):
    if stopwords:
        df['num_chars_q1_sw'] = df['question1'].apply(lambda x: sum([len([c for c in w if c not in punctuation]) for 
                                                                 w in x.strip().split() if
                                                                 w not in stopwords]))
        return(df)
    else:
        df['num_chars_q1'] = df['question1'].apply(lambda x: len(list([c for c in x.strip() if
                                                                       c not in punctuation])))
        return(df)


def num_chars_q2(df, stopwords = None):
    if stopwords:
        df['num_chars_q2_sw'] = df['question2'].apply(lambda x: sum([len([c for c in w if c not in punctuation]) for 
                                                                 w in x.strip().split() if
                                                                 w not in stopwords]))
        return(df)
    else:
        df['num_chars_q2'] = df['question2'].apply(lambda x: len(list([c for c in x.strip() if
                                                                       c not in punctuation])))
        return(df)


# difference in the number of characters in each question. 
# Assumes `num_chars_q1` and `num_chars_q2` were already called. 
def num_chars_diff (df, stopwords = None):
    if stopwords:
        df['num_chars_diff_sw'] = abs(df['num_chars_q1_sw'] - df['num_chars_q2_sw'])
        return(df)
    else:
        df['num_chars_diff'] = abs(df['num_chars_q1'] - df['num_chars_q2'])
        return(df)



####################################
# String edit distance features
# These use the `fuzzywuzzy` package to compute variations of Levenshtein and string distance features. 
####################################

# Partial word overlap. This will capture if the questions start the same way
def partial_ratio(df, stopwords = None):
    if stopwords:
        df['partial_ratio_sw'] = df.apply(lambda x: fuzz.partial_ratio(" ".join(
                    [w for w in x['question1'].lower().strip().split() if w not in punctuation and w not in stopwords]),
                    " ".join([w for w in x['question2'].lower().strip().split() if w not in punctuation and w not in stopwords])
                    ), axis = 1)
        return(df)
    else: 
        df['partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio("".join([c for c in x['question1'].lower().strip() if
                                                   c not in punctuation]), "".join([c for c in x['question2'].lower().strip() if
                                                   c not in punctuation])), axis = 1)
        return(df)


# partial token set ratio
def partial_token_set_ratio(df, stopwords = None):
    if stopwords:
        df['partial_token_set_ratio_sw'] = df.apply(lambda x: fuzz.partial_token_set_ratio(" ".join(
                    [w for w in x['question1'].lower().strip().split() if w not in punctuation and w not in stopwords]),
                    " ".join([w for w in x['question2'].lower().strip().split() if w not in punctuation and w not in stopwords])
                    ), axis = 1)
        return(df)
    else: 
        df['partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio("".join([c for c in x['question1'].lower().strip() if
                                                   c not in punctuation]), "".join([c for c in x['question2'].lower().strip() if
                                                   c not in punctuation])), axis = 1)
        return(df)


# partial token set ratio
def token_set_ratio(df, stopwords = None):
    if stopwords:
        df['token_set_ratio_sw'] = df.apply(lambda x: fuzz.token_set_ratio(" ".join(
                    [w for w in x['question1'].lower().strip().split() if w not in punctuation and w not in stopwords]),
                    " ".join([w for w in x['question2'].lower().strip().split() if w not in punctuation and w not in stopwords])
                    ), axis = 1)
        return(df)
    else: 
        df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio("".join([c for c in x['question1'].lower().strip() if
                                                   c not in punctuation]), "".join([c for c in x['question2'].lower().strip() if
                                                   c not in punctuation])), axis = 1)
        return(df)



# partial token sort ratio
def partial_token_sort_ratio(df, stopwords = None):
    if stopwords:
        df['partial_token_sort_ratio_sw'] = df.apply(lambda x: fuzz.partial_token_set_ratio(" ".join(
                    [w for w in x['question1'].lower().strip().split() if w not in punctuation and w not in stopwords]),
                    " ".join([w for w in x['question2'].lower().strip().split() if w not in punctuation and w not in stopwords])
                    ), axis = 1)
        return(df)
    else: 
        df['partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio("".join([c for c in x['question1'].lower().strip() if
                                                   c not in punctuation]), "".join([c for c in x['question2'].lower().strip() if
                                                   c not in punctuation])), axis = 1)
        return(df)


# token sort ratio
def token_sort_ratio(df, stopwords = None):
    if stopwords:
        df['token_sort_ratio_sw'] = df.apply(lambda x: fuzz.token_sort_ratio(" ".join(
                    [w for w in x['question1'].lower().strip().split() if w not in punctuation and w not in stopwords]),
                    " ".join([w for w in x['question2'].lower().strip().split() if w not in punctuation and w not in stopwords])
                    ), axis = 1)
        return(df)
    else: 
        df['token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio("".join([c for c in x['question1'].lower().strip() if
                                                   c not in punctuation]), "".join([c for c in x['question2'].lower().strip() if
                                                   c not in punctuation])), axis = 1)
        return(df)



# token sort ratio
def wratio(df, stopwords = None):
    if stopwords:
        df['wratio_sw'] = df.apply(lambda x: fuzz.WRatio(" ".join(
                    [w for w in x['question1'].lower().strip().split() if w not in punctuation and w not in stopwords]),
                    " ".join([w for w in x['question2'].lower().strip().split() if w not in punctuation and w not in stopwords])
                    ), axis = 1)
        return(df)
    else: 
        df['wratio'] = df.apply(lambda x: fuzz.WRatio("".join([c for c in x['question1'].lower().strip() if
                                                   c not in punctuation]), "".join([c for c in x['question2'].lower().strip() if
                                                   c not in punctuation])), axis = 1)
        return(df)




####################################
# Tf-Idf Features
# These compte features using Tf-Idf scores, 
# as well as compare word embeddings - weighted by Tf-Idf scores
####################################


# extract Tf-Idf weights for the dataset in question
def extract_tfidf(df):
    values = df['question1'].append(df['question2']).apply(lambda x: re.sub('[ ]{2,}', ' ',
                                        re.sub('[^a-z]', ' ', x.lower().strip()))).values
    # train tf-idf weights to the training quesitons
    vectorizer = TfidfVectorizer(stop_words = stopwords, lowercase = True)
    vectorizer.fit(values)

    # store the idf and tf-idf weights of each word in a dictionary
    idf = vectorizer._tfidf.idf_
    tfidf_weights = dict(zip(vectorizer.get_feature_names(), idf))

    # return both
    return(vectorizer, idf, tfidf_weights)


# Total tf-idf weight in words of question intersection and symmetric difference. 
# By default remove stopwords
def tf_idf_weight(df,vectorizer, stopwords = stopwords, norm = 'l2' ):


    # get the total tfidf weight for the words in the intersection of the two sentences. 
    df['word_intersection_tfidf_weight'] = df.apply(lambda x: vectorizer.transform([
            " ".join(set(x['question1'].split()).intersection(x['question2'].split()))]).sum(), axis = 1)
    
    # get the total tfidf weight for the words in the symmetric difference of the two sentences 
    df['word_symmetric_difference_tfidf_weight'] = df.apply(lambda x: vectorizer.transform([
            " ".join(set(x['question1'].split()).symmetric_difference(x['question2'].split()))]).sum(), axis = 1)
    
    return(df)



####################################
# Word embedding features
# These use various vector distance functions to compute similarities between words in each question
####################################


# get embedding
def get_embeddings():
    # load the pretrained word2vec embeddings 
    model = KeyedVectors.load_word2vec_format('~/Desktop/quora/data/embeddings/wiki.en.vec')
    return (model)

# A function for computing average embedding vector of words in questions
def sent2vec(s, model,  stopwords = stopwords, punctuation = punctuation):
    words = [w for w in s.lower().strip() if w not in stopwords and w not in punctuation and w.isalpha()]
    # collect all embeddings of the words in the sentence
    M = []
    for w in words:
        M.append(model[w])
    # Take the average embedding position
    M = np.array(M)
    # average over the columns to get an embedding of the sentence
    v = M.mean(axis = 0)
    # if failed, return origin
    if np.all(np.isnan(v)):
        return np.repeat(0, 300)
    return(v)

# get the sentence embeddings of the first question
def q1_embedding(df, model, stopwords = stopwords, punctuation = punctuation):
    # create a column for embedding of q1
    df['q1_embedding'] = [sent2vec(s, model = model, stopwords = stopwords, punctuation = punctuation) for
                          s in df['question1']]
    return(df)

# get the sentence embeddings of the second question
def q2_embedding(df, model, stopwords = stopwords, punctuation = punctuation):
    # create a column for embedding of q1
    df['q2_embedding'] = [sent2vec(s, model = model, stopwords = stopwords, punctuation = punctuation) for
                          s in df['question2']]
    return(df)

# get the word with the highest tf_idf weight in the question for question1 and question2
def q1_highest_tfidf_weight(df, tfidf_weights):
    def max_word(x):
        try:
            return(max([(w, tfidf_weights[w]) for w in x.strip().lower().split() if 
                                         w in tfidf_weights], key=itemgetter(1))[0])
        except:
            return("")
            
    df['q1_highest_tfidf_weight'] = df['question1'].apply(lambda x: max_word(x))
    return(df)

# get the word with the highest tf_idf weight in the question for question1 and question2
def q2_highest_tfidf_weight(df, tfidf_weights):
    def max_word(x):
        try:
            return(max([(w, tfidf_weights[w]) for w in x.strip().lower().split() if 
                                         w in tfidf_weights], key=itemgetter(1))[0])
        except:
            return("")
    df['q2_highest_tfidf_weight'] = df['question2'].apply(lambda x: max_word(x))
    return(df)

# Embedding of word with highest tf-idf weight in first question
def q1_max_tf_idf_embedding(df, model):
    df['q1_max_tf_idf_embedding'] = [model[w] if w in model else np.repeat(0,300) for w in df['q1_highest_tfidf_weight']]
    return(df)

# Embedding of word with highest tf-idf weight in second question
def q2_max_tf_idf_embedding(df, model):
    df['q2_max_tf_idf_embedding'] = [model[w] if w in model else np.repeat(0,300) for w in df['q2_highest_tfidf_weight'] ]
    return(df)

# Euclidean distance between sentence embeddings
def euclidean_distance_sentence_embeddings(df):
    df['euclidean_distance_sentence_embeddings'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(df['q1_embedding']),
                                                          np.nan_to_num(df['q2_embedding']))]
    return(df)


# Euclidean distance between words of highest tf-Idf weight
def euclidean_distance_max_tfidf_word(df):
    df['euclidean_distance_max_tfidf_word'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(df['q1_max_tf_idf_embedding']),
                                                          np.nan_to_num(df['q2_max_tf_idf_embedding']))]
    return(df)

# Cosine distance between sentence embeddings
def cosine_distance_sentence_embeddings(df):
    df['cosine_distance_sentence_embeddings'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(df['q1_embedding']),
                                                          np.nan_to_num(df['q2_embedding']))]
    
    # fill NaN values as discussed below
    fill = df[pd.isnull(df['cosine_distance_sentence_embeddings'])].apply(lambda x: 0 if
                        np.array_equal(x['q1_embedding'], x['q2_embedding'])  
                        else 1, axis = 1)
    
    if len(fill) > 0:
        df['cosine_distance_sentence_embeddings'] = df['cosine_distance_sentence_embeddings'].fillna(fill)
        
    return(df)


# cosine distance of words in each question with maximum tf-idf weight
def cosine_distance_max_tfidf_word(df):
    df['cosine_distance_max_tfidf_word'] = [cosine(x, y) for (x, y) in zip(df['q1_max_tf_idf_embedding'],
                                                          df['q2_max_tf_idf_embedding'])]
    
    # fill NaN values as discussed above
    fill = df[pd.isnull(df['cosine_distance_max_tfidf_word'])].apply(lambda x: 0 if
                        np.array_equal(x['q1_max_tf_idf_embedding'], x['q2_max_tf_idf_embedding'])  
                        else 1, axis = 1)
    
    if len(fill) > 0:
        df['cosine_distance_max_tfidf_word'] = df['cosine_distance_max_tfidf_word'].fillna(fill)
    return(df, fill)


# Manhattan distance between sentence embeddings
def cityblock_distance_sentence_embeddings(df):
    df['cityblock_distance_sentence_embeddings'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(df['q1_embedding']),
                                                          np.nan_to_num(df['q2_embedding']))]
    return(df)

# Jaccard distance between sentence embeddings
def jaccard_distance_sentence_embeddings(df):
    df['jaccard_distance_sentence_embeddings'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(df['q1_embedding']),
                                                          np.nan_to_num(df['q2_embedding']))]
    return(df)

# Brraycurtis distance between sentence embeddings
def braycurtis_distance_sentence_embeddings(df):
    df['braycurtis_distance_sentence_embeddings'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(df['q1_embedding']),
                                                          np.nan_to_num(df['q2_embedding']))]
    return(df)



####################################
# Calculate all features
# Takes in a pandas dataframe, calls all these functions above, and augments the dataframe before returning it.
####################################



def extract_all_features(df):

    # get a word embedding library
    model = get_embeddings()

    # get a tf-idf vectorizer, idf weights and tf-idf weights
    (vectorizer, idf, tfidf_weights) = extract_tfidf(df)


    '''
    Extract syntactic features
    '''
    # number of words in common
    df = len_intersection(df)
    # number of words in common (excluding stopwords)
    df = len_intersection(df, stopwords = stopwords)
    # number of words in q1
    df = num_words_q1(df)
    # number of words in q1 (excluding stopwords)
    df = num_words_q1(df, stopwords = stopwords)
    # number of words in q2 
    df = num_words_q2(df)
    # number of words in q2 (excluding stopwords)
    df = num_words_q2(df, stopwords = stopwords)
    # difference in the number of words
    df = num_words_diff(df)
    # difference in the number of words (excluding stopwords)
    df = num_words_diff(df, stopwords = stopwords)
    # number of characters in q1
    df = num_chars_q1(df)
    # number of characters in q1 (excluding stopwords)
    df = num_chars_q1(df, stopwords = stopwords)
    # number of characters in q2 
    df = num_chars_q2(df)
    # number of characters in q2 (excluding stopwords)
    df = num_chars_q2(df, stopwords = stopwords)
    # difference in number of characters
    df = num_chars_diff(df)
    # difference in number of characters(excluding stopwords)
    df = num_chars_diff(df, stopwords = stopwords)

    '''
    Extract string distance features
    '''
    # partial ratio between sentences
    df = partial_ratio(df)
    # partial ratio between sentences (excluding stopwords)
    df = partial_ratio(df, stopwords = stopwords)
    # partial token ratio between sentences
    df = partial_token_set_ratio(df)
    # partial token ratio between sentences (excluding stopwords)
    df = partial_token_set_ratio(df, stopwords = stopwords)
    # token set ratio
    df = token_set_ratio(df)
    # token set ratio (stopwords excluded)
    df = token_set_ratio(df, stopwords = stopwords)
    # partial token sort ratio 
    df = partial_token_sort_ratio(df)
    # partial token sort ratio (stopwords excluded)
    df = partial_token_sort_ratio(df, stopwords = stopwords)
    # token sort ratio
    df = token_set_ratio(df)
    # token sort ratio (stopwords excluded)
    df = token_set_ratio(df, stopwords = stopwords)
    # w-ratio
    df = wratio(df)
    # w-ratio (stopwords excluded)
    df = wratio(df, stopwords = stopwords)
    '''
    Tf-idf features
    '''
    # total tf-idf weight in words in sentence (excluding stopwords)
    df = tf_idf_weight(df, vectorizer = vectorizer)
    '''
    Word embedding features
    '''
    # sentence embedding of quesiton1 (stopwords excluded)
    df = q1_embedding(df, model = model)
    # sentence embedding of question2 (stopwords excluded)
    df = q2_embedding(df, model = model)
    # euclidean distance between sentence embeddings
    df = euclidean_distance_sentence_embeddings(df)
    # cosine distance between sentence embeddings
    df = cosine_distance_sentence_embeddings(df)
    # cityblock distance between sentence embeddings
    df = cityblock_distance_sentence_embeddings(df)
    # Jaccard distance between sentence embeddings
    df = jaccard_distance_sentence_embeddings(df)
    # braycurtis distance between sentence embeddings
    df = braycurtis_distance_sentence_embeddings(df)
    # word in question1 with highest tfidf weight
    df = q1_highest_tfidf_weight(df, tfidf_weights = tfidf_weights)
    # word in question2 with highest tfidf weight
    df = q2_highest_tfidf_weight(df, tfidf_weights = tfidf_weights)
    # embedding of word in question1 with highest tfidf weight
    df = q1_max_tf_idf_embedding(df, model = model)
    # embedding of word in question2 with highest tfidf weight
    df = q2_max_tf_idf_embedding(df, model = model)
    # euclidean distance between embedding of words of highest tfidf weight
    df = euclidean_distance_max_tfidf_word(df)
    # cosine distance between embeddings of words of highest tfidf wieght
    df = cosine_distance_max_tfidf_word(df)


    return(df)









































