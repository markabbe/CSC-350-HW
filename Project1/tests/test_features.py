import pytest
from pytest import approx

from oswegonlp import preprocessing, classifier_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logistic_regression, features
import numpy as np
import torch

def setup_module():
    global vocab, label_set, x_tr_pruned, X_tr

    y_tr,x_tr = preprocessing.read_data('fakenews-train.csv',preprocessor=preprocessing.bag_of_words)
    labels = set(y_tr)

    counts_tr = preprocessing.aggregate_word_counts(x_tr)

    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr, x_tr, 5)

    X_tr = preprocessing.make_numpy(x_tr_pruned,vocab)
    label_set = sorted(list(set(y_tr)))

def test_d6_1_topfeat_numpy():
    top_feats_fake = features.get_top_features_for_label_numpy(hand_weights.theta_manual,'fake',3)
    assert (top_feats_fake[0] == (('fake', 'media'), 0.2))
    assert (len(top_feats_fake) == 3)
    
    top_feats_real = features.get_top_features_for_label_numpy(hand_weights.theta_manual,'real',3)
    assert (top_feats_real[0] == (('real', 'signs'), 0.3))
    assert (len(top_feats_real) == 3)

def test_d6_2_topfeat_torch():
	global vocab, label_set
	model_test = torch.load('tests/test_weights.torch')

	top_feats_two = features.get_top_features_for_label_torch(model_test, vocab, label_set,'real',5)
	assert (top_feats_two == ['north', 'says', 'us', 'donald', 'trumps'])

	top_feats_nine = features.get_top_features_for_label_torch(model_test, vocab, label_set,'fake',7)
	assert (top_feats_nine == ['you', 'is', 'and', 'just', 'a', 'the', 'hillary'])
