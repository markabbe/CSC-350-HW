import pytest
from pytest import approx

from oswegonlp.constants import START_TAG, END_TAG, TRAIN_FILE
from oswegonlp import hmm, most_common
import numpy as np

def setup_module():
    global tag_trans_counts, hmm_trans_weights
    tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
    

# 4.1a
def test_tag_trans_counts():
    global tag_trans_counts
    
    assert (tag_trans_counts['DET']['NOUN'] == 9671)
    assert (tag_trans_counts[START_TAG]['NOUN'] == 782)
    assert (tag_trans_counts[START_TAG]['PUNCT'] == 433)