import pytest
import oswegonlp
from pytest import approx
from oswegonlp.constants import TRAIN_FILE, DEV_FILE
from oswegonlp import most_common, classifier_base, preprocessing, scorer, tagger_base

#1.1a (0.5 points)
def test_get_top_noun_tags():
    expected = [('time', 385), ('people', 233), ('way', 187)]
    tag_word_counts = most_common.get_tag_word_counts(TRAIN_FILE)
    actual = tag_word_counts["NOUN"].most_common(3)
    assert expected == actual

#1.1b
def test_get_top_verb_tags():
    expected = [('have', 749), ('get', 359), ('know', 338)]
    tag_word_counts = most_common.get_tag_word_counts(TRAIN_FILE)
    actual = tag_word_counts["VERB"].most_common(3)
    assert expected == actual