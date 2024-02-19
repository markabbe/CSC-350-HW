import pytest
from pytest import approx

from oswegonlp import preprocessing, classifier_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logistic_regression, features
import numpy as np
import torch

def setup_module():
    global vocab, label_set, x_tr_pruned, X_tr, Y_tr

    y_tr,x_tr = preprocessing.read_data('fakenews-train.csv',preprocessor=preprocessing.bag_of_words)
    labels = set(y_tr)

    counts_tr = preprocessing.aggregate_word_counts(x_tr)

    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr, x_tr, 5)

    label_set = sorted(list(set(y_tr)))
    X_tr = preprocessing.make_numpy(x_tr_pruned,vocab)
    Y_tr = np.array([label_set.index(y_i) for y_i in y_tr])

def test_d6_1_topfeat_numpy():
    top_feats_fake = features.get_top_features_for_label_numpy(hand_weights.theta_manual,'fake',3)
    assert (top_feats_fake[0] == (('fake', 'media'), 0.2))
    assert (len(top_feats_fake) == 3)
    
    top_feats_real = features.get_top_features_for_label_numpy(hand_weights.theta_manual,'real',3)
    assert (top_feats_real[0] == (('real', 'signs'), 0.3))
    assert (len(top_feats_real) == 3)

def test_d6_2_topfeat_torch():
    global vocab, label_set
    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))

    model.eval()  

    top_feats_real = features.get_top_features_for_label_torch(model, vocab, label_set,'real',10)
    top_feats_fake = features.get_top_features_for_label_torch(model, vocab, label_set,'fake',10)
    
    top_feats_real_sorted = sorted(top_feats_real)
    top_feats_fake_sorted = sorted(top_feats_fake)
    
    expected_real_features_sorted = sorted(['australia', 'turnbull', 'travel', 'korea', 'ban', 'north', 'says', 'us', 'donald', 'trumps'])
    expected_fake_features_sorted = sorted(['that', 'of', 'it', 'you', 'is', 'and', 'just', 'a', 'the', 'hillary'])

    assert top_feats_real_sorted == expected_real_features_sorted, f"Expected features for 'real' not found. Got {top_feats_real}, expected {expected_real_features_sorted}"
    assert top_feats_fake_sorted == expected_fake_features_sorted, f"Expected features for 'fake' not found. Got {top_feats_fake}, expected {expected_fake_features_sorted}"
