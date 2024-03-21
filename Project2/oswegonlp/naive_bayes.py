from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation, preprocessing

import numpy as np
import itertools
import functools
from collections import defaultdict
from collections import Counter
import math

def get_nb_weights(trainfile, smoothing):
    """
    estimate_nb function assumes that the labels are one for each document, where as in POS tagging: we have labels for 
    each particular token. So, in order to calculate the emission score weights: P(w|y) for a particular word and a 
    token, we slightly modify the input such that we consider each token and its tag to be a document and a label. 
    The following helper code converts the dataset to token level bag-of-words feature vector and labels. 
    The weights obtained from here will be used later as emission scores for the viterbi tagger.
    
    inputs: train_file: input file to obtain the nb_weights from
    smoothing: value of smoothing for the naive_bayes weights
    
    :returns: nb_weights: naive bayes weights
    """
    token_level_docs=[]
    token_level_tags=[]
    for words,tags in preprocessing.conll_seq_generator(trainfile):
        token_level_docs += [{word:1} for word in words]
        token_level_tags +=tags
    nb_weights = estimate_nb(token_level_docs, token_level_tags, smoothing)
    
    return nb_weights

def corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    label_counts = defaultdict(Counter)

    for features, instance_label in zip(x, y):
        if instance_label == label:
            label_counts[instance_label].update(features)

    return label_counts[label]

# deliverable 3.2
def estimate_pxy(x,y,label,alpha,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param alpha: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    label_corpus_counts = corpus_counts(x, y, label)
    
    total_word_count = sum(label_corpus_counts.values()) + alpha * len(vocab)
    
    log_probabilities = defaultdict(float)
    
    for word in vocab:
        word_count = label_corpus_counts[word] + alpha
        log_probabilities[word] = math.log(word_count / total_word_count)
    
    return log_probabilities

# deliverable 3.3
def estimate_nb(x,y,alpha):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    
    vocab = set(feature for sample in x for feature in sample)
    label_counts = Counter(y)
    feature_counts = {label: Counter() for label in label_counts}
    
    #update feature counts with the features in each sample
    for features, label in zip(x, y):
        feature_counts[label].update(features)
    
    weights = defaultdict(float)

    #calculate the weights for each label
    for label in label_counts:
        total_count = sum(feature_counts[label].values())
        denominator = total_count + alpha * len(vocab)
        #calculate the smoothed log prob for each feature, given the label
        for feature in vocab:
            smoothed_count = feature_counts[label][feature] + alpha
            weights[(label, feature)] = math.log(smoothed_count) - math.log(denominator)
        
        weights[(label, '**OFFSET**')] = math.log(label_counts[label]) - math.log(len(y))

    return weights

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,alphas):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param alphas: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    best_alpha = None
    best_accuracy = 0
    accuracy_scores = {}

    for alpha in alphas:
        theta_nb = estimate_nb(x_tr, y_tr, alpha)
        
        y_hat_dv = [classifier_base.predict(x, theta_nb, ['real', 'fake'])[0] for x in x_dv]

        accuracy = evaluation.acc(y_hat_dv, y_dv)
        
        accuracy_scores[alpha] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    return best_alpha, accuracy_scores
    







