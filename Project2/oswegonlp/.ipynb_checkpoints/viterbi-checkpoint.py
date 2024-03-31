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

# OMG IT WORKS
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

    # Had to extend some variable names to help figure things out LOL
    backpointers = []
    viterbivars=[]
    
    # Iterate over each possible current tag
    for curr_tag in list(all_tags):
        # List to store scores for all possible previous tags transitioning to current tag
        temp_scores = []
        for prev_tag in list(all_tags):
            # If no transition possible
            if prev_tag == END_TAG or curr_tag == START_TAG:
                tempScore = torch.tensor(-np.inf)
            else:
                # Else, calculate indices for current and previous tags
                current_index = tag_to_ix[curr_tag]
                prev_index = tag_to_ix[prev_tag]
                tempScore = prev_scores[0][prev_index] + transition_scores[current_index][prev_index] + cur_tag_scores[current_index]
            temp_scores.append(tempScore)
        backpointers.append(temp_scores.index(max(temp_scores)))
        viterbivars.append(max(temp_scores))
            
    return viterbivars, backpointers
        

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
    
    tag_index_map = {value: key for key, value in tag_to_ix.items()}
    
    # Initialize the start vector with negative infinity for all tags except the START_TAG, which is set to 0.
    start_vector = np.full((1, len(all_tags)), -np.inf)
    start_vector[0][tag_to_ix[START_TAG]] = 0
    # Convert the start vector to a PyTorch Variable for compatibility
    previous_scores = torch.autograd.Variable(torch.from_numpy(start_vector.astype(np.float32))).view(1, -1)
    # Empty list to hold backpointers for each step in the sequence
    backpointers_list = []
    
    # Iterate over each set of current tag scores for each token in the sequence
    for step_index in range(len(cur_tag_scores)):
        # Calculate the Viterbi scores and backpointers for the current step
        step_scores, step_backpointers = viterbi_step(all_tags, tag_to_ix, cur_tag_scores[step_index], transition_scores, previous_scores)
    
        # Extract the scores from the step_scores tensor as a list and update the previous_scores with the current step values  
        step_values = [step_scores[score_index].item() for score_index in range(len(step_scores))]
        previous_scores = torch.autograd.Variable(torch.from_numpy(np.asarray([step_values])))
    
        backpointers_list.append(step_backpointers)
    
    # Prepare the end vector with negative infinity for all positions and set the END_TAG position to 0
    end_vector = np.full((1, len(all_tags)), -np.inf)
    end_vector[0][tag_to_ix[END_TAG]] = 0
    final_scores = torch.from_numpy(end_vector)
    last_step_scores = final_scores[0]
    
    # Perform the final Viterbi step with the end vector to complete the trellis
    last_word_scores, last_word_backpointers = viterbi_step(all_tags, tag_to_ix, last_step_scores, transition_scores, previous_scores)
    # Find the index with the highest score in the last step
    final_index = np.argmax(last_word_scores)
    # Extract the highest score as the path score
    highest_score = last_word_scores[final_index]
    # Use the final index to find the corresponding tag for the last backpointer
    final_backpointer = tag_index_map[last_word_backpointers[final_index]]
    # Append the last word's backpointer to the list
    backpointers_list.append(last_word_backpointers)
    
    path_index = len(backpointers_list) - 1
    path_score = highest_score
    path_tags = []
    # Backtrack through the backpointer list to construct the best path
    while(path_index != 0):
        path_tags.append(tag_index_map[backpointers_list[path_index][final_index]])
        final_index = backpointers_list[path_index][final_index]
        path_index -= 1
    
    best_path = path_tags[::-1]
    
    return path_score, best_path
