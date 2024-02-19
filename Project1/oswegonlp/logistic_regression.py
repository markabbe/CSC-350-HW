from oswegonlp import evaluation
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim

# deliverable 5.2
def build_linear(x, y):
    '''
    Build a linear model in torch

    :param x: the set of training documents
    :param y: the set of training labels
    :returns: PyTorch linear model
    :rtype: PyTorch model
    '''
    size1 = x.shape[1]
    size2 = len(set(y))

    model = torch.nn.Sequential()
    model.add_module("Linear",torch.nn.Linear(size1, size2, bias = True))

    return model

# deliverable 5.3
def log_softmax(scores):
    '''
    Compute the (log of the) softmax given the scores

    Hint: Use logsumexp

    :param scores: output of linear model as a numpy array
    :returns: the softmax result
    :rtype: numpy array
    '''
    c = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - c)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    softmax = exp_scores / sum_exp_scores
    log_softmax = np.log(softmax)
    return log_softmax

# deliverable 5.4
def nll_loss(logp, y):
    '''
    Compute the neg-log likelihood loss from log softmax probabilities, averaged across documents

    return the loss in a number
    :param logp: output of log softmax
    :param y: the set of training labels
    :returns: the NLL loss
    :rtype: float
    '''

    nlls = 0

    for row,y_i in zip(logp,y):
        nlls += -row[y_i]

    return nlls / len(y)


######################### helper code
def train_model(loss, model, X_tr_var, Y_tr_var,
                num_its=200,
                X_dv_var=None,
                Y_dv_var=None,
                status_frequency=10,
                optim_args={'lr':0.002,'momentum':0},
                param_file='best.params'):
    
    #initialize the optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)
    
    losses = []
    accuracies = []

    best_accuracy = 0
    
    #training loop over number of iterations
    for epoch in range(num_its):
        #set gradient to zero
        optimizer.zero_grad()
        output = model(X_tr_var)
        loss_val = loss(output, Y_tr_var)
        #backpropagate and train
        loss_val.backward()
        optimizer.step()

        losses.append(loss_val.item())

        #write parameters if this is the best epoch yet
        if X_dv_var is not None and Y_dv_var is not None:
            #run forward on dev data
            Y_hat_dev = model(X_dv_var)
            _, Y_hat = Y_hat_dev.max(dim=1)
            #calculate the accuracy and append to the list
            acc = evaluation.acc(Y_hat.data.numpy(), Y_dv_var.data.numpy())
            accuracies.append(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                state = {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': acc
                }
                torch.save(state, param_file)
                
                save_path = 'competition-dev.preds.npy'
                np.save(save_path, Y_hat.data.numpy())
                print(f"Saved best model predictions to '{save_path}' at epoch {epoch+1}")

        if epoch % status_frequency == 0 or epoch == num_its - 1:
            print(f"Epoch {epoch+1}/{num_its}: Loss {loss_val.item():.4f} Acc {acc:.4f}")

    if os.path.isfile(param_file):
        print("Loading best model parameters from file.")
        checkpoint = torch.load(param_file)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Best model loaded from {param_file} with accuracy {best_accuracy:.4f}.")

        best_preds_path = 'competition-dev.preds.npy'
        if os.path.isfile(best_preds_path):
            best_preds = np.load(best_preds_path)
            print(f"Loaded best model predictions from '{best_preds_path}'.")

    return model, losses, accuracies


def plot_results(losses, accuracies):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies);
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');
