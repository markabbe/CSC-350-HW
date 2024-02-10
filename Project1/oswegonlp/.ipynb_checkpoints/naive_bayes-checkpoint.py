import functools
import itertools

from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation

import numpy as np
from collections import defaultdict, Counter
import math

# deliverable 3.1
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
    
    for features, label in zip(x, y):
        feature_counts[label].update(features)
    
    weights = defaultdict(float)
    for label in label_counts:
        total_count = sum(feature_counts[label].values())
        denominator = total_count + alpha * len(vocab)
        for feature in vocab:
            smoothed_count = feature_counts[label][feature] + alpha
            weights[(label, feature)] = math.log(smoothed_count) - math.log(denominator)
        
        weights[(label, '**OFFSET**')] = math.log(label_counts[label]) - math.log(len(y))
    
    print("Label counts:", label_counts)
    for label in label_counts:
        print(f"Weights for label {label}:")
        for feature in vocab:
            print(f"  {feature}: {weights[(label, feature)]}")
        print(f"  OFFSET: {weights[(label, '**OFFSET**')]}")
    
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

    # Assuming labels are 'real' and 'fake'
    labels = ['real', 'fake']

    for alpha in alphas:
        # Estimate the model using the current alpha
        theta_nb = estimate_nb(x_tr, y_tr, alpha)
        
        # Use the predict function to predict labels for the development set
        y_hat_dv = [predict(x, theta_nb, labels)[0] for x in x_dv]

        # Calculate accuracy for the development set predictions
        accuracy = evaluation.acc(y_hat_dv, y_dv)
        
        accuracy_scores[alpha] = accuracy

        # Update best alpha if the current one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    return best_alpha, accuracy_scores





