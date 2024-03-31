from oswegonlp.constants import OFFSET
import numpy as np
import operator

# use this to find the highest-scoring label
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# deliverable 2.1
def make_feature_vector(x, y):
    """
    Take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param x: counter of base features
    :param y: label string
    :returns: dict of features, f(x,y)
    :rtype: dict
    """
    feature_vector = {}

    feature_vector[(y, '**OFFSET**')] = 1

    for feature_name, count in x.items():
        feature_vector[(y, feature_name)] = count

    return feature_vector

# deliverable 2.2
def predict(x, weights, labels):
    """
    Prediction function
    :param x: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict
    """
    scores = {label: 0 for label in labels}
    
    #compute score for each label
    for label in labels:
        score = 0
        #add the score of each feature in the feature vector
        for feature, value in x.items():
            if (label, feature) in weights:
                score += weights[(label, feature)] * value
        if (label, '**OFFSET**') in weights:
            score += weights[(label, '**OFFSET**')]
        scores[label] = score

    #determine label with max score
    top_label = max(scores, key=scores.get)
    
    return top_label, scores


