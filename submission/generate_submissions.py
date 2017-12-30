import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from operator import itemgetter
from scipy.spatial.distance import cosine, cityblock, jaccard, euclidean, braycurtis
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import pickle as pkl

from basic_features import extract_all_features
from semantic_distance import extract_semantnic_similarity
from question_classification_predictions import predict_question_class

import time

import pdb


if __name__ == '__main__':

    # time execution
    start_time = time.time()

    # load data
    print
    print("Loading test set...")
    df = pd.read_csv("../data/test.csv")
    df.fillna("oov", inplace = True)
    print("done.")
    print


    # # extract basic features
    print("Extracting basic features...")
    df = extract_all_features(df)[0]
    print("done.")

    # extract semantic similarity scores
    print("Extracting semantic similarity scores...")
    df = extract_semantnic_similarity(df)
    print("done.")

    # extract question class predictions
    print("Extracting question class predictions...")
    df = predict_question_class(df)
    print("done.")
    print

    # # load the pre-trained Gradient Boosted Trees model
    with open("../models/xgb_full_model.pickle", "rb") as handle:
        xgb_full = pkl.load(handle)


    # isolate features to be used to make predictions
    test_features = [u'len_intersection',
    u'num_words_q1',
    u'num_words_q2',
    u'num_chars_q1',
    u'num_chars_q2',
    u'num_chars_diff',
    u'partial_ratio',
    u'partial_ratio_sw',
    u'token_set_ratio',
    u'token_set_ratio_sw',
    u'partial_token_sort_ratio_sw',
    u'wratio_sw',
    u'word_intersection_tfidf_weight',
    u'word_symmetric_difference_tfidf_weight',
    u'euclidean_distance_sentence_embeddings',
    u'cosine_distance_sentence_embeddings',
    u'cityblock_distance_sentence_embeddings',
    u'braycurtis_distance_sentence_embeddings',
    u'euclidean_distance_max_tfidf_word',
    u'cosine_distance_max_tfidf_word',
    u'lch_similarity', 
    u'embedding_similarity_score', 
    u'lstm_vote_q1',
    u'lstm_vote_q2', 
    u'lstm_vote_agree']



    # add predictions
    print("Classifying question pairs...")
    df['is_duplicate'] = xgb_full.predict(df[test_features])
    print("done.")
    print

    # keep only the test_id and response columns
    df = df[['test_id', 'is_duplicate']]

    # # sort table based on test_id
    df = df.sort('test_id')


    # # write submission
    print("Writing submissions...")
    df.to_csv("submission.csv", index = False)
    print("done.")
    print

    # print execution time
    print("--- %s seconds ---" % (time.time() - start_time))


