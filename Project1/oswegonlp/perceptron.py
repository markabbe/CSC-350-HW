from collections import defaultdict
from oswegonlp.classifier_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''

    #make the prediction using the current weights
    y_hat, _ = predict(x, weights, labels) 

    #if the prediction is incorrect, perform the update
    if y != y_hat:
        correct_features = make_feature_vector(x, y)
        predicted_features = make_feature_vector(x, y_hat)

        #initialize the updates as a defaultdict
        updates = defaultdict(float)

        #add the correct features to the update
        for feature, value in correct_features.items():
            updates[feature] += value

        #subtract the predicted features from the update
        for feature, value in predicted_features.items():
            updates[feature] -= value

        return updates

    #if the prediction is correct, no update is needed
    return defaultdict(float)

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            y_hat, _ = predict(x_i, weights, labels)
            if y_i != y_hat:
                update = perceptron_update(x_i, y_i, weights, labels)
                for feature, value in update.items():
                    weights[feature] += value 
        weight_history.append(weights.copy())
    return weights, weight_history

