import pytest
from pytest import approx

from oswegonlp.constants import START_TAG, END_TAG, TRAIN_FILE
from oswegonlp import hmm, most_common
import numpy as np

def setup():
    global tag_trans_counts, hmm_trans_weights
    tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
    hmm_trans_weights = hmm.compute_transition_weights(tag_trans_counts,.001)


# 4.2a
def test_hmm_trans_weights_sum_to_one():
    global tag_trans_counts, hmm_trans_weights

    all_tags = list(tag_trans_counts.keys()) + [END_TAG]
    for tag in tag_trans_counts.keys():
        assert sum(np.exp(hmm_trans_weights[(next_tag,'NOUN')]) for next_tag in all_tags) == approx(1,abs=1e-5)

# 4.2b
def test_hmm_trans_weights_exact_vals():
    global hmm_trans_weights

    assert hmm_trans_weights[('NOUN',START_TAG)] == approx(-2.77506,abs=1e-3)
    assert hmm_trans_weights[('VERB',START_TAG)] == approx(-2.79835,abs=1e-3)
    assert hmm_trans_weights[('INTJ','DET')] == approx(-16.60569,abs=1e-3)
    assert hmm_trans_weights[('NOUN','DET')] == approx(-0.52105,abs=1e-3)
    assert (hmm_trans_weights[(START_TAG,'VERB')] == -np.inf)
    

    
