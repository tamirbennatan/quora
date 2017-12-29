import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from operator import itemgetter
from scipy.spatial.distance import cosine, cityblock, jaccard, euclidean, braycurtis
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords

from basic_features import extract_all_features
from semantic_distance import extract_sematnic_similarity

import pdb

# load sample
df = pd.read_csv("../data/sample.csv")


# extract all the features
df = extract_all_features(df)

sample.to_csv("../data/sample_basic_extracted.csv")