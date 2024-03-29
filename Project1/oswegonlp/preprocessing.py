from collections import Counter

import numpy as np
import pandas as pd

from oswegonlp.re_tokenizer import RegexTokenizer

# deliverable 1.1
def bag_of_words(text, retok):
    '''
    Count the number of word occurences for each document in the corpus

    :param retok: an instance of RegexTokenizer
    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    
    tokens = retok.tokenize(text)

    return Counter(tokens)

# deliverable 1.2
def aggregate_word_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counter = Counter()
    for bag in bags_of_words:
        counter.update(bag)
    return counter
    
# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''

    distinctBow1 = set()
    for word in bow1:
        if word not in (bow2):
            distinctBow1.add(word)

    return distinctBow1
    
# deliverable 1.4
def prune_vocabulary(training_counts,target_data,threshold):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    vocab = {word for word, count in training_counts.items() if count >= threshold}
    
    x_pruned = [Counter({word: count for word, count in text.items() if word in vocab}) for text in target_data]
    
    return x_pruned, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab_size = len(vocab)
    
    sorted_vocab = sorted(vocab)
    
    word_to_index = {word: i for i, word in enumerate(sorted_vocab)}
    
    data_array = np.zeros((len(bags_of_words), vocab_size))
    
    for i, bow in enumerate(bags_of_words):
        for word, count in bow.items():
            if word in word_to_index: 
                data_array[i, word_to_index[word]] = count
    
    return data_array
    
### Helper Code ###

def read_data(filename,label='RealOrFake',preprocessor=bag_of_words):
    retok = RegexTokenizer("[A-Za-z']+")
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string,retok) for string in df['Headline'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())