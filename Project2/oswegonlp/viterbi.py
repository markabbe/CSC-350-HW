import operator
from collections import defaultdict, Counter
from oswegonlp.constants import START_TAG,END_TAG, UNK
import numpy as np
import torch
import torch.nn
from torch import autograd
from torch.autograd import Variable

def get_torch_variable(arr):
    # returns a pytorch variable of the array
    torch_var = torch.autograd.Variable(torch.from_numpy(np.array(arr).astype(np.float32)))
    return torch_var.view(1,-1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def viterbi_step(all_tags, tag_to_ix, cur_tag_scores, transition_scores, prev_scores):
    """
    Calculates the best path score and corresponding back pointer for each tag for a word in the sentence in pytorch, which you will call from the main viterbi routine.
    
    parameters:
    - all_tags: list of all tags: includes both the START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    - cur_tag_scores: pytorch Variable that contains the local emission score for each tag for the current token in the sentence
                       it's size is : [ len(all_tags) ] 
    - transition_scores: pytorch Variable that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    - prev_scores: pytorch Variable that contains the scores for each tag for the previous token in the sentence: 
                    it's size is : [ 1 x len(all_tags) ] 
    
    :returns:
    - viterbivars: a list of pytorch Variables such that each element contains the score for each tag in all_tags for the current token in the sentence
    - bptrs: a list of idx that contains the best_previous_tag for each tag in all_tags for the current token in the sentence
    """
    bptrs = []
    viterbivars = []

    # Additional debugging to inspect the specific tag scores
    print(f"Emission probabilities for NOUN: {cur_tag_scores[tag_to_ix['NOUN']]}")
    print(f"Transition scores to NOUN: {transition_scores[:, tag_to_ix['NOUN']]}")
    print(f"Previous scores: {prev_scores}")

    for next_tag in all_tags:
        next_tag_ix = tag_to_ix[next_tag]
        next_tag_var = prev_scores + transition_scores[:, next_tag_ix].view(1, -1) + cur_tag_scores[next_tag_ix].view(1, -1)
        best_tag_id = argmax(next_tag_var)
        bptrs.append(best_tag_id)
        viterbivars.append(next_tag_var[0][best_tag_id].view(1, -1))

    viterbivars = torch.cat(viterbivars, dim=1)

    noun_tag_index = tag_to_ix['NOUN']
    noun_score = viterbivars[0, noun_tag_index].item()
    print(f"Score for NOUN tag: {noun_score}")

    # Ensure the assertion is informative
    assert noun_score == -2, f"Expected score for NOUN is -2, got {noun_score}"

    return viterbivars, bptrs

def build_trellis(all_tags, tag_to_ix, cur_tag_scores, transition_scores):
    """
    This function should compute the best_path and the path_score. 
    Use viterbi_step to implement build_trellis in viterbi.py in Pytorch.
    
    parameters:
    - all_tags: a list of all tags: includes START_TAG and END_TAG
    - tag_to_ix: a dictionary that maps each tag to a unique id.
    - cur_tag_scores: a list of pytorch Variables where each contains the local emission score for each tag for that particular token in the sentence, len(cur_tag_scores) will be equal to len(words)
                        it's size is : [ len(words in sequence) x len(all_tags) ] 
    - transition_scores: pytorch Variable (a matrix) that contains the tag_transition_scores
                        it's size is : [ len(all_tags) x len(all_tags) ] 
    
    :returns:
    - path_score: the score for the best_path
    - best_path: the actual best_path, which is the list of tags for each token: exclude the START_TAG and END_TAG here.
    """
    
    # Create the inverse mapping from index to tag name
    ix_to_tag = {index: tag for tag, index in tag_to_ix.items()}
    
    # Initialize the list to store the back pointers
    whole_bptrs = []

    # Initialize the previous scores with the score for the START_TAG
    initial_vec = np.full((1, len(all_tags)), -np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0
    prev_scores = get_torch_variable(initial_vec)
    
    # Iterate over the sentence
    for step, cur_tag_score in enumerate(cur_tag_scores):
        # At each step, calculate Viterbi variables and backpointers.
        viterbivars_t, bptrs_t = viterbi_step(all_tags, tag_to_ix, cur_tag_score, transition_scores, prev_scores)
        whole_bptrs.append(bptrs_t)
        # Update scores for the next step
        prev_scores = viterbivars_t.view(1, -1)

    # Transition to the END_TAG
    terminal_var = prev_scores + transition_scores[tag_to_ix[END_TAG]]
    terminal_var[0][tag_to_ix[START_TAG]] = -np.inf
    path_score, best_tag_id = torch.max(terminal_var, 1)
    best_path = [best_tag_id.item()]

    # Follow the back pointers to decode the best path
    for bptrs_t in reversed(whole_bptrs):
        best_tag_id = bptrs_t[best_tag_id]
        if isinstance(best_tag_id, int):
            best_path.append(best_tag_id)
        else:
            best_path.append(best_tag_id.item())

    # Remove the start tag from the path and reverse the path
    start = best_path.pop()
    best_path.reverse()
    
    assert start == tag_to_ix[START_TAG]
    best_path = [ix_to_tag[ix] for ix in best_path]
    
    return path_score, best_path

