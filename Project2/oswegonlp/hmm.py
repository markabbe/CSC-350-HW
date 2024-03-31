from oswegonlp.preprocessing import conll_seq_generator
from oswegonlp.constants import START_TAG, END_TAG, OFFSET, UNK
from oswegonlp import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable
from collections import defaultdict, Counter


def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    weights = defaultdict(float)
    all_tags = list(trans_counts.keys())+ [END_TAG]

    # Set the transition count from START_TAG to END_TAG to 0 to avoid illegal transitions
    trans_counts[START_TAG][END_TAG] = 0

    # Iterate through all pairs of tags to compute transition weights
    for prev_tag in all_tags:
        for curr_tag in all_tags:
            # If transitioning from the start tag or to the end tag, set the weight to negative infinity
            if curr_tag == START_TAG or prev_tag == END_TAG:
                weights[(curr_tag, prev_tag)] = -np.inf
            else:
                # Compute the numerator as the log of the transition count from prev_tag to curr_tag, smoothed
                numerator = np.log(smoothing + trans_counts[prev_tag][curr_tag])
                # Compute the denominator as the log of the total count of transitions out of prev_tag, smoothed
                denominator = np.log((len(all_tags) - 1) * smoothing + sum(trans_counts[prev_tag].values()))
                # Calculate the weight (log probability) for the transition from prev_tag to curr_tag
                weights[(curr_tag, prev_tag)] = numerator - denominator
    return weights


def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Initialize tag transition probabilities with -np.inf for impossible transitions
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)

    # Initialize emission probabilities with -np.inf to indicate impossible emissions by default
    emission_probs = np.full((len(vocab), len(tag_to_ix)), -np.inf)

    # Fill in tag transition probabilities
    for (curr_tag, prev_tag), log_prob in hmm_trans_weights.items():
        if curr_tag in tag_to_ix and prev_tag in tag_to_ix:
            tag_transition_probs[tag_to_ix[curr_tag], tag_to_ix[prev_tag]] = log_prob

    # Initialize emission_probs for words in vocab according to nb_weights
    for (tag, word), log_prob in nb_weights.items():
        if word in word_to_ix and tag in tag_to_ix:
            emission_probs[word_to_ix[word], tag_to_ix[tag]] = log_prob

    # Convert to PyTorch Variables
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))

    return emission_probs_vr, tag_transition_probs_vr
    
