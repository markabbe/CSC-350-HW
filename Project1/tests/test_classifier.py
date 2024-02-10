from oswegonlp import preprocessing, classifier_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logistic_regression

import numpy as np
import pytest

from pytest import approx

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

    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr, x_tr, 10)
    x_dv_pruned, _ = preprocessing.prune_vocabulary(counts_tr, x_dv, 10)


def test_d2_1_featvec():
    label = 'fake'
    fv = classifier_base.make_feature_vector({'test':1,'case':2},label)
    assert len(fv) == 3
    assert (fv[(label,'test')] == 1)
    assert (fv[(label,'case')] == 2)
    assert (fv[(label,constants.OFFSET)] == 1)

def test_d2_2_predict():
    global x_tr_pruned, x_dv_pruned, y_dv

    y_hat,scores = classifier_base.predict(x_tr_pruned[58],hand_weights.theta_manual,labels)
    assert scores['fake'] == 0.1
    assert scores['real'] == 0.01
    assert y_hat == 'fake'

    y_hat = classifier_base.predict_all(x_dv_pruned,hand_weights.theta_manual,labels)
    assert evaluation.acc(y_hat,y_dv) == approx(.6554878048780488, abs=1e-5)

def test_d3_1_corpus_counts():
    # public
    iama_counts = naive_bayes.corpus_counts(x_tr_pruned,y_tr,"fake");
    assert iama_counts['news'] == 25
    assert iama_counts['tweet'] == 5
    assert iama_counts['internets'] == 0


def test_d3_2_pxy():
    global vocab, x_tr_pruned, y_tr
    
    # check that distribution normalizes to one
    log_pxy = naive_bayes.estimate_pxy(x_tr_pruned,y_tr,"fake",0.1,vocab)
    assert np.exp(list(log_pxy.values())).sum() == approx(1)

    # check that values are correct
    log_pxy['media'] == approx(-5.650512594555945,abs=1e-3)
    log_pxy['hillary'] == approx(-4.0980803192446675,abs=1e-3)

    log_pxy_more_smooth = naive_bayes.estimate_pxy(x_tr_pruned,y_tr,"fake",10,vocab)
    log_pxy_more_smooth['media'] == approx(-5.692404863096082,abs=1e-3)
    log_pxy_more_smooth['hillary'] == approx(-4.409788931451033,abs=1e-3)

def test_d3_3a_nb():
    global x_tr_pruned, y_tr

    theta_nb = naive_bayes.estimate_nb(x_tr_pruned,y_tr,0.1)

    y_hat,scores = classifier_base.predict(x_tr_pruned[55],theta_nb,labels)
    assert scores['fake'] == approx(-24.615867370043578,abs=1e-3)
    assert  y_hat == 'real'

    y_hat,scores = classifier_base.predict(x_tr_pruned[155],theta_nb,labels)
    assert scores['real'] == approx(-21.0895135009066, abs=1e-3)
    assert y_hat == 'real'

def test_d3_3b_nb():
    global y_dv
    y_hat_dv = evaluation.read_predictions('nb-dev.preds')
    assert evaluation.acc(y_hat_dv,y_dv) >= .8

def test_d3_4a_nb_best():
    global x_tr_pruned, y_tr, x_dv_pruned, y_dv
    vals = np.logspace(-3,2,11)
    best_smoother, scores = naive_bayes.find_best_smoother(x_tr_pruned,y_tr,x_dv_pruned,y_dv,[1e-3,1e-2,1e-1,1])
    assert scores[.1] >= .81
    assert scores[.01] >= .82



