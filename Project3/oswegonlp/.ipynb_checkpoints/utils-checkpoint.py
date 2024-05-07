import torch
import torch.autograd as ag
from oswegonlp.constants import END_OF_INPUT_TOK, HAVE_CUDA

import numpy as np

if HAVE_CUDA:
    import torch.cuda as cuda

def to_scalar(var):
    """
    Wrap up the terse, obnoxious code to go from torch.Tensor to
    a python int / float (is there a better way?)
    """
    if isinstance(var, ag.Variable):
        return var.data.view(-1).tolist()[0]
    else:
        return var.view(-1).tolist()[0]


def argmax(vector):
    """
    Takes in a row vector (1xn) and returns its argmax
    """
    _, idx = torch.max(vector, 1)
    return to_scalar(idx)


def initialize_with_pretrained(pretrained_embeds, word_embedding):
    """
    Initialize the embedding lookup table of word_embedding with the embeddings
    from pretrained_embeds.
    Remember that word_embedding has a word_to_ix member you will have to use.
    For every word that we do not have a pretrained embedding for, keep the default initialization.
    :param pretrained_embeds dict mapping word to python list of floats (the embedding
        of that word)
    :param word_embedding The network component to initialize (i.e, a VanillaWordEmbedding
        or BiLSTMWordEmbedding)
    """
    # Get the embedding layer from wor embedding object
    embed_from_word = word_embedding.word_embeddings
    
    # Get the dictionary mapping words to their indices
    word_idx_map = word_embedding.word_to_ix
    
    # Initialize a counter for tracking the number of updated embeddings
    updated_embeddings_count = 1
    
    # Iterate through each word in the word index mapping
    for word in word_idx_map.keys():
        # Check if the current word has a pretrained embedding
        if word in pretrained_embeds.keys():
            # Get the index of the word
            idx = word_idx_map[word]
            
            # Increment the counter for each update
            updated_embeddings_count += 1
            
            # Update embedding layer with the pretrained embedding for word
            embed_from_word.weight.data[idx] = torch.Tensor(pretrained_embeds[word])

def build_suff_to_ix(word_to_ix):
    """
        From a word to index vocab lookup, create a suffix-to-index vocab lookup
        For our purposes, a suffix consists of just the last two letters of a word.
            If the word is a single letter, just use the whole word
        :param word_to_ix: the vocab as a dict
        :return suff_to_ix: the suffix lookup as a dict
    """
    # Initialize a set to store unique suffixes or whole words
    suffix_set = set()
    
    # Iterate through each word in the dictionary keys
    for word in word_to_ix.keys():
        word_length = len(word)  # Calculate the length of the word
        
        # If the word is shorter than 2 characters, add the entire word to the set
        if word_length < 2:
            suffix_set.add(word)
        else:
            # Extract the last two characters as the suffix and add to the set
            suffix = word[-2:word_length]
            suffix_set.add(suffix)
    
    # Create a dictionary mapping each unique suffix to an index
    suffix_to_index = {suffix: index for index, suffix in enumerate(sorted(suffix_set))}
    
    return suffix_to_index


# ===----------------------------------------------------------------===
# Dummy classes that let us test parsing logic without having the
# necessary components implemented yet
# ===----------------------------------------------------------------===
class DummyCombiner:

    def __call__(self, head, modifier):
        return head


class DummyActionChooser:

    def __init__(self):
        self.counter = 0

    def __call__(self, inputs):
        self.counter += 1
        return ag.Variable(torch.Tensor([0., 0., 1.]))


class DummyWordEmbedding:

    def __init__(self):
        self.word_embeddings = lambda x: None
        self.counter = 0

    def __call__(self, sentence):
        self.counter += 1
        return [None]*len(sentence)


class DummyFeatureExtractor:

    def __init__(self):
        self.counter = 0

    def get_features(self, parser_state):
        self.counter += 1
        return []


