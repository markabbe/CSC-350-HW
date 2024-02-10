import pytest
from pytest import approx

from oswegonlp import preprocessing, classifier_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logistic_regression
import numpy as np
from collections import Counter

def setup_module():
    #global y_tr, x_tr, corpus_counts, labels, vocab
    #corpus_counts = get_corpus_counts(x_tr)


    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned, x_bl_pruned
    global labels
    global vocab

    y_tr,x_tr = preprocessing.read_data('fakenews-train.csv',preprocessor=preprocessing.bag_of_words)
    labels = set(y_tr)

    counts_tr = preprocessing.aggregate_word_counts(x_tr)

    y_dv,x_dv = preprocessing.read_data('fakenews-dev.csv',preprocessor=preprocessing.bag_of_words)

    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr, x_tr, 5)
    x_dv_pruned, _ = preprocessing.prune_vocabulary(counts_tr, x_dv, 5)

def test_d4_1_perc_update():
    global x_tr_pruned, y_tr

    labels = set(y_tr)

    theta_perc = Counter()
    update = perceptron.perceptron_update(x_tr_pruned[1500],y_tr[1500],theta_perc,labels)
    assert (len(update) == 0)

    update = perceptron.perceptron_update(x_tr_pruned[1],y_tr[1],theta_perc,labels)
    assert (len(update) == 18)
    assert (update[('real','trump')] == 1)
    assert (update[('fake','son')] == -1)
    assert (update[('real',constants.OFFSET)] == 1)
    assert (update[('fake',constants.OFFSET)] == -1)

def test_d4_2a_perc_estimate():
    global y_dv, x_tr_pruned, y_tr

    # run on a subset of data
    theta_perc,theta_perc_history = perceptron.estimate_perceptron(x_tr_pruned[1800:1900],y_tr[1800:1900],3)
    assert (theta_perc[('fake','trump')] == -1)
    assert (theta_perc[('fake','clinton')] == 1)
    assert (theta_perc[('real','claims')] == -1)
    assert (theta_perc[('real','about')] == 2)
    assert (theta_perc_history[0][('real','about')] == 1)
    

def test_d4_2b_perc_accuracy():
    global y_dv
    # i get 43% accuracy
    y_hat_dv = evaluation.read_predictions('perc-dev.preds')
    assert (evaluation.acc(y_hat_dv,y_dv) >= .7)

