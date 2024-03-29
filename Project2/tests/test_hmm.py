import pytest
from pytest import approx

from oswegonlp.constants import * 
from oswegonlp import hmm, viterbi, most_common, scorer, naive_bayes
import numpy as np

def setup_module():
    global nb_weights, hmm_trans_weights, all_tags
    nb_weights = naive_bayes.get_nb_weights(TRAIN_FILE, .01)
    tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
    hmm_trans_weights = hmm.compute_transition_weights(tag_trans_counts,.01)
    all_tags = list(tag_trans_counts.keys()) + [END_TAG]
    

# 4.2
def test_hmm_on_example_sentence():
    global nb_weights, hmm_trans_weights, all_tags
    tag_to_ix={}
    for tag in list(all_tags):
        tag_to_ix[tag]=len(tag_to_ix)
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE)
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights, \
                                                                         vocab, word_to_ix, tag_to_ix)
    
    score, pred_tags = viterbi.build_trellis(all_tags,
                                             tag_to_ix,
                                             [emission_probs[word_to_ix[w]] for w in ['they', 'can', 'can', 'fish','.']],
                                             tag_transition_probs)
    
    assert score.item() == approx(-28.5981, abs=1e-2)
    assert (pred_tags == ['PRON', 'AUX', 'AUX', 'ADJ','PUNCT'])

# 4.4a
def test_hmm_dev_accuracy():
    confusion = scorer.get_confusion(DEV_FILE,'hmm-dev-en.preds')
    acc = scorer.accuracy(confusion)
    assert (acc > .870)

# 4.4b
def test_hmm_test_accuracy():
    confusion = scorer.get_confusion(TEST_FILE,'hmm-te-en.preds')
    acc = scorer.accuracy(confusion)
    assert (acc > .880)

# 4.5a
def test_nr_hmm_dev_accuracy():
    confusion = scorer.get_confusion(NR_DEV_FILE,'hmm-dev-nr.preds')
    acc = scorer.accuracy(confusion)
    assert (acc > .910)

# 4.5b
def test_nr_hmm_test_accuracy():
    confusion = scorer.get_confusion(NR_TEST_FILE,'hmm-te-nr.preds')
    acc = scorer.accuracy(confusion)
    assert (acc > .903)


