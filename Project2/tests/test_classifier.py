import pytest
from pytest import approx
from oswegonlp.constants import TRAIN_FILE, DEV_FILE
from oswegonlp import most_common, classifier_base, preprocessing, scorer, tagger_base

def setup():
    global all_tags, theta_mc, tagger_mc

    all_tags = preprocessing.get_all_tags(TRAIN_FILE)

    theta_mc = most_common.get_most_common_word_weights(TRAIN_FILE)
    tagger_mc = tagger_base.make_classifier_tagger(theta_mc)

    theta_mc = most_common.get_most_common_word_weights(TRAIN_FILE)

## when there are multiple tests for a single question, must pass *both* tests for credit

#2.2a 
def test_mcc_tagger_output():
    global tagger_mc, all_tags
    
    tags = tagger_mc(['They','can','can','fish'],all_tags)
    assert (tags == ['PRON','AUX','AUX','NOUN'])

    tags = tagger_mc(['The','old','man','the','boat','.'],all_tags)
    assert (tags == ['DET', 'ADJ', 'NOUN', 'DET', 'NOUN', 'PUNCT'])
        
#2.2b
def test_mcc_tagger_accuracy():
    global tagger_mc, all_tags
        
    expected = 0.838369

    confusion = tagger_base.eval_tagger(tagger_mc,'most-common.preds',all_tags=all_tags)
    actual = scorer.accuracy(confusion)
    
    assert expected < actual
