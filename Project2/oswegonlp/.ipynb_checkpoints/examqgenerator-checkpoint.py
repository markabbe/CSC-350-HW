from constants import * 
import hmm, viterbi, most_common, scorer, naive_bayes
import numpy as np

nb_weights = naive_bayes.get_nb_weights(TRAIN_FILE, .01)
tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
hmm_trans_weights = hmm.compute_transition_weights(tag_trans_counts,.01)
all_tags = list(tag_trans_counts.keys()) + [END_TAG]

def run_sentence(sentence_toks):
    global nb_weights, hmm_trans_weights, all_tags
    tag_to_ix={}
    for tag in list(all_tags):
        tag_to_ix[tag]=len(tag_to_ix)
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE)
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights, \
                                                                         vocab, word_to_ix, tag_to_ix)
    
    score, pred_tags = viterbi.build_trellis(all_tags,
                                             tag_to_ix,
                                             [emission_probs[word_to_ix[w]] for w in sentence_toks],
                                             tag_transition_probs)
    
    return pred_tags