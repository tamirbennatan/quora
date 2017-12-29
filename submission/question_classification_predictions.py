'''
Extract predictions of question answer type
LSTM's trained and saved in `question_classification/trec_lstms.ipynb`
LSTM's trained on TREC dataset, annotated by Li and Roth (2004)
'''

import pandas as pd
import pickle as pkl
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def get_tokenizer():
    with open("../models/tokenizer.pickle", "rb") as handle:
        tokenizer = pkl.load(handle)
    return(tokenizer)

# A function which takes in a numpy array of quesitions (strings)
# and returns padded index vectors usable by deep learning models. 
def encode_and_pad(questions, tokenizer, sequence_length = 10):
    # questions encoded as index vectors
    encoded = tokenizer.texts_to_sequences(questions)
    # padded squences to be of length [sequence_length]
    padded = pad_sequences(encoded, 
                            maxlen = sequence_length,
                            padding = "post", 
                            truncating = "post")
    return(padded)


# return the most frequently occuring value in an array
def most_common(lst):
    return max(set(lst), key=lst.count)


def predict_question_class(df):

    # Load the trained LSTM models
    trec_lstm1 = load_model("../models/trec_lstm1.h5")
    trec_lstm2 = load_model("../models/trec_lstm2.h5")
    trec_lstm3 = load_model("../models/trec_lstm3.h5")
    trec_lstm4 = load_model("../models/trec_lstm4.h5")
    trec_lstm5 = load_model("../models/trec_lstm5.h5")

    # get the pre-trained word tokenizer 
    tokenizer = get_tokenizer()

    # make predictions for the question classes using loaded models

    # add predictions of first model, first question
    df['lstm_1_q1_pred'] = trec_lstm1.predict_classes(
            encode_and_pad(df['question1'].values, tokenizer)
            )
    # add predictions of first model, first question
    df['lstm_1_q2_pred'] = trec_lstm1.predict_classes(
            encode_and_pad(df['question2'].values, tokenizer)
        )

    # add predictions of second model, first question
    df['lstm_2_q1_pred'] = trec_lstm2.predict_classes(
            encode_and_pad(df['question1'].values, tokenizer)
        )

    # add predictions of second model, second question
    df['lstm_2_q2_pred'] = trec_lstm2.predict_classes(
            encode_and_pad(df['question2'].values, tokenizer)
        )

    # add predictions of third model, first question
    df['lstm_3_q1_pred'] = trec_lstm3.predict_classes(
            encode_and_pad(df['question1'].values, tokenizer)
        )

    # add predictions of third model, second question
    df['lstm_3_q2_pred'] = trec_lstm3.predict_classes(
            encode_and_pad(df['question2'].values, tokenizer)
        )

    # add predictions of fourth model, first question
    df['lstm_4_q1_pred'] = trec_lstm4.predict_classes(
            encode_and_pad(df['question1'].values, tokenizer)
        )

    # add predictions of fourth model, second question
    df['lstm_4_q2_pred'] = trec_lstm4.predict_classes(
            encode_and_pad(df['question2'].values, tokenizer)
        )

    # add predictions of fifth model, second question
    df['lstm_5_q1_pred'] = trec_lstm5.predict_classes(
            encode_and_pad(df['question1'].values, tokenizer, sequence_length = 6)
        )

    # add predictions of fifth model, second question
    df['lstm_5_q2_pred'] = trec_lstm5.predict_classes(
            encode_and_pad(df['question2'].values, tokenizer, sequence_length = 6)
        )

    # most commonly predicted class of question1 amongst the five LSTM models
    df['lstm_vote_q1'] = df.apply(lambda x: \
                 most_common([x['lstm_1_q1_pred'], x['lstm_2_q1_pred'], \
                              x['lstm_3_q1_pred'], x['lstm_4_q1_pred'], \
                              x['lstm_5_q1_pred']]), axis = 1)

    # most commonly predicted class of queston2 amongst the five LSTM models
    df['lstm_vote_q2'] = df.apply(lambda x: \
                 most_common([x['lstm_1_q2_pred'], x['lstm_2_q2_pred'], \
                              x['lstm_3_q2_pred'], x['lstm_4_q2_pred'], \
                              x['lstm_5_q2_pred']]), axis = 1)

    # A feature which shows if the most vote classes agree
    df['lstm_vote_agree'] = (df['lstm_vote_q1'] == df['lstm_vote_q2'])

    return(df)



