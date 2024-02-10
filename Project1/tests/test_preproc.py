from oswegonlp import preprocessing

import pytest

def setup_module():
    global x_tr, y_tr, x_dv, y_dv, counts_tr, counts_dv, counts_bl, x_dv_pruned, x_tr_pruned
    global vocab
    y_tr,x_tr = preprocessing.read_data('fakenews-train.csv',preprocessor=preprocessing.bag_of_words)
    y_dv,x_dv = preprocessing.read_data('fakenews-dev.csv',preprocessor=preprocessing.bag_of_words)

    counts_tr = preprocessing.aggregate_word_counts(x_tr)
    counts_dv = preprocessing.aggregate_word_counts(x_dv)

def test_d1_1_bow():
    global x_tr, y_tr
    assert (len(x_tr) == len(y_tr))
    assert (x_tr[4]['phone'] == 1)
    assert (x_tr[41]['to'] == 3)
    assert (x_tr[410]['to'] == 0)
    assert (len(x_tr[1000]) == 9)

def test_d1_2_agg():
    global x_dv

    assert (counts_dv['donald'] == 103)
    assert (len(counts_dv) == 1322)
    assert (counts_dv['to'] == 97)

def test_d1_3a_oov():
    global counts_tr, counts_dv
    assert (len(preprocessing.compute_oov(counts_dv,counts_tr)) == 322)
    assert (len(preprocessing.compute_oov(counts_tr,counts_dv)) == 3739)


def test_d1_4_prune():
    global x_dv, counts_tr

    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr,x_tr,3)
    x_dv_pruned, vocab2 = preprocessing.prune_vocabulary(counts_tr,x_dv,3)

    assert (len(vocab) == 1309)

    assert (len(x_dv[95].keys())-len(x_dv_pruned[95].keys()) == 1)
    

