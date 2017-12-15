import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Merge

from sklearn.model_selection import train_test_split

from gensim.models import KeyedVectors

import pickle as pkl




# --------------------------------------------
# Directory specific parameters (where to find stuff)
# --------------------------------------------

# Where to find "train.csv"
DATA_FILE = "../data/processed/train.csv"
# Where to find the pre-trained tokenizer
TOKENIZER_FILE = "../models/tokenizer.pickle"
# where to find the fasttext embeddings
EMBEDDING_FILE = "../data/embeddings/wiki.en.vec"
# where to save the file once you're done training
SAVE_DEST = "../models/siamese_lstm.h5"



# --------------------------------------------
# Encoding and embedding utilites
# --------------------------------------------


''''
Text tokenizer. 
It is important to use my pre-trained tokenizer, 
to keep sentence encodings consistent.
'''

# load pickled word tokenzier
with open(TOKENIZER_FILE, "rb") as handle:
    tokenizer = pkl.load(handle)

# The number of types in the joint datset
vocab_size = len(tokenizer.word_index) + 1

'''
A function which takes in a numpy array of quesitions (strings)
and returns padded index vectors usable by deep learning models. 
'''
def encode_and_pad(questions, sequence_length = 25):
    # questions encoded as index vectors
    encoded = tokenizer.texts_to_sequences(questions)
    # padded squences to be of length [sequence_length]
    padded = pad_sequences(encoded, 
                            maxlen = sequence_length,
                            padding = "post", 
                            truncating = "post")
    return(padded)



# load the pretrained fasttext embeddings (this takes a while)
print("loading word embeddings...")

embedding_model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE)

# Each row in the matrix is the embedding of one word in the dataset. 
# The row index corresponds to the integer ecoding of that word. 
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    if word in embedding_model:
        embedding_matrix[i] = embedding_model[word]

print("done loading word embeddings.")
print

# --------------------------------------------
# Setting up test environment
# --------------------------------------------

# Split data into training and test sets.

print("Splittin data into train/dev/test splits...")

X_train_and_dev, X_test, y_train_and_dev, y_test = train_test_split(
    data[data.columns - ["is_duplicate"]], data['is_duplicate'], \
    test_size=0.2, random_state=550)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_and_dev, y_train_and_dev, test_size=0.25, random_state=550)

print("done.")
print

# --------------------------------------------
# Model specific parameters
# --------------------------------------------

N_HIDDEN_UNITS = 60
N_EPOCHS = 25
BATCH_SIZE = 64 #(6317 updates per epoch)
SEQUENCE_LENGTH = 30


# --------------------------------------------
# Model archetecutre
# --------------------------------------------


# The function with which I will merge the layers
# Credit Elior Cohen https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


# Define the input units - one for each question
input1 = Input(shape=(SEQUENCE_LENGTH,), name="Question1-Input")
input2 = Input(shape=(SEQUENCE_LENGTH,), name="Question2-Input")

# add Embedding layer on top of first input 
embedding1 = Embedding(input_dim = vocab_size, 
                     output_dim = 300, 
                     input_length = SEQUENCE_LENGTH,
                     trainable = False)(input1)

# add Embedding layer on top of second input
embedding2 = Embedding(input_dim = vocab_size, 
                     output_dim = 300, 
                     input_length = SEQUENCE_LENGTH,
                     trainable = False)(input2)

# a shared LSTM unit. This is what makes it a Siamese LSTM
shared_lstm = LSTM(N_HIDDEN_UNITS)



# get the outpouts of the two inputs, applied to the same LSTM unit
left_output = shared_lstm(embedding1)
right_output = shared_lstm(embedding2)

# Merge the two outputs using Manhattan similarity 
merged = Merge(mode=lambda x:\
               exponent_neg_manhattan_distance(x[0], x[1]), \
               output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# instantiate model
MaLSTM = Model([input1, input2], [merged])



# --------------------------------------------
# Compile and Train!
# --------------------------------------------

MaLSTM.compile(loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['acc'])

# For consic
print("BEGIN TRAINING")
print
trained = MaLSTM.fit([encode_and_pad(question1_train, SEQUENCE_LENGTH), encode_and_pad(question2_train, SEQUENCE_LENGTH)], y_train,
           batch_size = BATCH_SIZE, nb_epoch = N_EPOCHS, \
           validation_data = ([encode_and_pad(question1_dev, SEQUENCE_LENGTH), encode_and_pad(question2_dev, SEQUENCE_LENGTH)], y_dev))

# save model weights
MaLSTM.save(SAVE_DEST)

print
print("Completed and saved to %s" % SAVE_DEST)
