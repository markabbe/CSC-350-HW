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
    label_features_weights = [((label, feature), weight) for (feature_label, feature), weight in weights.items() if feature_label == label]

    label_features_weights.sort(key=lambda x: x[1], reverse=True)

    top_features = label_features_weights[:k]

    return top_features

# deliverable 6.2
#cant figure out why this is failing
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
    print("Original vocab size:", len(vocab))
    vocab = sorted(vocab)
    print("Sorted vocab size:", len(vocab))

    label_index = label_set.index(label)
    print("Label index for '{}': {}".format(label, label_index))

    linear_weights = model.linear.weight.data if hasattr(model, 'linear') else model[0].weight.data
    print("Linear weights shape:", linear_weights.shape)

    label_weights = linear_weights[label_index]
    print("Label weights shape:", label_weights.shape)

    if not isinstance(label_weights, np.ndarray):
        label_weights = label_weights.cpu().numpy()
        print("Converted label weights to numpy.")

    feature_weights = list(zip(vocab, label_weights))
    print("Feature weights length:", len(feature_weights))

    sorted_features = sorted(feature_weights, key=lambda x: x[1], reverse=True)
    print("Sorted features length:", len(sorted_features))

    top_features = [feature for feature, weight in sorted_features[:k]]
    print("Top features:", top_features)

    return top_features