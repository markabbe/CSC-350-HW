from oswegonlp.constants import OFFSET
from collections import Counter
from torch.autograd import Variable
import numpy as np
import torch

# deliverable 6.1
def get_top_features_for_label_numpy(weights,label,k=5):
    '''
    Return the k features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''
    raise NotImplementedError


# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the k words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''
    vocab = sorted(vocab)

    raise NotImplementedError
