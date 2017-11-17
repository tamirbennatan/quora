"""
Functions for computing semantic similarity of sentences using WordNet similarity scores. 

- Original Paper which proposes this clas of metrics: https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf
- Much code taken from this blog post: http://nlpforhackers.io/wordnet-sentence-similarity/
"""

import nltk.wordnet as wn
from nltk import word_tokenize, pos_tag

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

"""
In the proposed algorithm, we should only compare words of the same POS category
(nouns, verbs, adjectives or adverbs). 

Below is a function which produces these broad POS tag categories (n, v, a, r)
from the Penn POS tags, which are more finely grained. 
"""
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
	# get the tag of that synset
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        # get the first synset of the word in WordNet, filtered by POS tag
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

# ---------------------------------------------------------------
# Similiarity scores
# ---------------------------------------------------------------
"""
A series of functions that take in two sentences (strings), 
and assign a similarity score based on WordNet distnace metrics. 

These similarity functions are symmetric.
For a more in depth explanation, see `wordnet_distance.ipynb` 
"""

# Distance betweeen synsets in WordNet - through the hyponym/hypernym path
def path_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets1])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    # Average the values and add score from both sides to get symmetic distance
    score = .5*(score1/count1 + score2/count2)
    return(score)


# Distance between words in WordNet graph, 
# weighted by depth of word in taxonomy. 
def lch_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
 
    # For each word in the first sentence
    for synset in synsets1:
        # isolate the part of speech 
        synset_pos = synset.pos()
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([synset.lch_similarity(ss) for ss in synsets2 if ss.pos() == synset_pos])
        except:
            best_score = None
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for synset in synsets2:
        # isolate the part of speech 
        synset_pos = synset.pos()
        try:
            # Get the similarity value of the most similar word in the other sentence
            best_score = max([synset.lch_similarity(ss) for ss in synsets1 if ss.pos() == synset_pos])
        except:
            best_score = None
            
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    # Average the values and add score from both sides to get symmetic score
    score = .5*(score1/count1 + score2/count2)
    return(score)

"""
a distance metric based on the depth of the two senses in the taxonomy 
and that of their Least Common Subsumer (most specific ancestor node)
"""
def wup_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score1, count1, score2, count2 = 0.0, 0, 0.0, 0 
 
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.wup_similarity(ss) for ss in synsets2])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score1 += best_score
            count1 += 1
            
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.wup_similarity(ss) for ss in synsets1])
 
        # Check that the similarity could have been computed
        if best_score is not None:
            score2 += best_score
            count2 += 1
 
    # Average the values and add score from both sides to get symmetic distance
    score = .5*(score1/count1 + score2/count2)
    return(score)




