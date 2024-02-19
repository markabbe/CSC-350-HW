import pytest
from pytest import approx
import os

from oswegonlp import preprocessing, classifier_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logistic_regression
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

def setup_module():
    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned
    global labels
    global vocab
    global X_tr, X_tr_var, X_dv_var, X_te_var, Y_tr, Y_dv, Y_tr_var, Y_dv_var, Y_te_var

    print("Current working directory:", os.getcwd())

    current_dir = os.path.dirname(__file__)
    print("Directory of the test script:", current_dir)

    #because for some reason i cant manually set the path to the .csv's
    train_path = os.path.join(current_dir, '..', 'fakenews-train.csv')
    dev_path = os.path.join(current_dir, '..', 'fakenews-dev.csv')
    test_path = os.path.join(current_dir, '..', 'fakenews-test-hidden.csv')

    print("Train path:", train_path)
    print("Dev path:", dev_path)
    print("Test path:", test_path)

    #please exist
    print("Does train_path exist?", os.path.exists(train_path))
    print("Does dev_path exist?", os.path.exists(dev_path))
    print("Does test_path exist?", os.path.exists(test_path))

    y_tr, x_tr = preprocessing.read_data(train_path, preprocessor=preprocessing.bag_of_words)
    labels = set(y_tr)

    counts_tr = preprocessing.aggregate_word_counts(x_tr)

    y_dv, x_dv = preprocessing.read_data(dev_path, preprocessor=preprocessing.bag_of_words)
    #y_te, x_te = preprocessing.read_data(test_path, preprocessor=preprocessing.bag_of_words)


    x_tr_pruned, vocab = preprocessing.prune_vocabulary(counts_tr, x_tr, 5)
    x_dv_pruned, _ = preprocessing.prune_vocabulary(counts_tr, x_dv, 5)
    #x_te_pruned, _ = preprocessing.prune_vocabulary(counts_tr,x_te,5)

    ## remove this, so people can run earlier tests  
    
    X_tr = preprocessing.make_numpy(x_tr_pruned,vocab)
    X_dv = preprocessing.make_numpy(x_dv_pruned,vocab)
    #X_te = preprocessing.make_numpy(x_te_pruned,vocab)
    label_set = sorted(list(set(y_tr)))
    Y_tr = np.array([label_set.index(y_i) for y_i in y_tr])
    Y_dv = np.array([label_set.index(y_i) for y_i in y_dv])
    #Y_te = np.array([label_set.index(y_i) for y_i in y_te])

    X_tr_var = Variable(torch.from_numpy(X_tr.astype(np.float32)))
    X_dv_var = Variable(torch.from_numpy(X_dv.astype(np.float32)))
    #X_te_var = Variable(torch.from_numpy(X_te.astype(np.float32)))

    Y_tr_var = Variable(torch.from_numpy(Y_tr))
    Y_dv_var = Variable(torch.from_numpy(Y_dv))     
    #Y_te_var = Variable(torch.from_numpy(Y_te))
    

def test_d5_1_numpy():
    global x_dv, counts_tr
    
    x_dv_pruned, vocab = preprocessing.prune_vocabulary(counts_tr,x_dv,10)
    X_dv = preprocessing.make_numpy(x_dv_pruned,vocab)

    print("Total sum of X_dv:", X_dv.sum())
    print("Sum of 5th row:", X_dv.sum(axis=1)[4])
    print("Sum of 145th row:", X_dv.sum(axis=1)[144])
    print("Sum of 11th column:", X_dv.sum(axis=0)[10])
    print("Sum of 101st column:", X_dv.sum(axis=0)[100])
    
    assert (X_dv.sum() == 1925)
    assert (X_dv.sum(axis=1)[4] == 4)
    assert (X_dv.sum(axis=1)[144] == 5)

    assert (X_dv.sum(axis=0)[10] == 4)
    assert (X_dv.sum(axis=0)[100] == 1)
    
def test_d5_2_logreg():
    global X_tr, Y_tr, X_dv_var

    model = logistic_regression.build_linear(X_tr,Y_tr)
    scores = model.forward(X_dv_var)
    assert (scores.size()[0] == 328)
    assert (scores.size()[1] == 2)


def test_d5_3_log_softmax():

    scores = np.asarray([[-0.1721,-0.5167,-0.2574,0.1571],[-0.3643,0.0312,-0.4181,0.4564]], dtype=np.float32)
    ans = logistic_regression.log_softmax(scores)
    assert ans[0][0] == approx(-1.3904355, abs=1e-5)
    assert ans[1][1] == approx(-1.3458145, abs=1e-5)
    assert ans[0][1] == approx(-1.7350391, abs=1e-5)


def test_d5_4_nll_loss():
    global X_tr, Y_tr, X_dv_var

    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()
    logP = model.forward(X_tr_var)
    assert logistic_regression.nll_loss(logP.data.numpy(), Y_tr) == approx(0.6929905418230031, abs=1e-5)

def test_d5_5_accuracy():
    global Y_dv_var

    #i think this is what im supposed to do?
    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()

    #train_model
    model_trained, losses, accuracies = logistic_regression.train_model(loss,model,
                                                       X_tr_var,
                                                       Y_tr_var,
                                                       X_dv_var=X_dv_var,
                                                       Y_dv_var = Y_dv_var,
                                                       num_its=400,
                                                       optim_args={'lr':0.2})
    
    acc = evaluation.acc(np.load('logreg-es-dev.preds.npy'),Y_dv_var.data.numpy())
    assert (acc >= 0.8)


def test_d7_3_competition_dev1():
    global Y_dv_var

    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()

    #train_model
    model_trained, losses, accuracies = logistic_regression.train_model(loss,model,
                                                       X_tr_var,
                                                       Y_tr_var,
                                                       X_dv_var=X_dv_var,
                                                       Y_dv_var = Y_dv_var,
                                                       num_its=900,
                                                       optim_args={'lr':0.5})
    
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert (acc >= 0.835)

def test_d7_3_competition_dev2():
    global Y_dv_var

    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()

    #train_model
    model_trained, losses, accuracies = logistic_regression.train_model(loss,model,
                                                       X_tr_var,
                                                       Y_tr_var,
                                                       X_dv_var=X_dv_var,
                                                       Y_dv_var = Y_dv_var,
                                                       num_its=900,
                                                       optim_args={'lr':0.5})
    
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert (acc >= 0.84)

def test_d7_3_competition_dev3():
    global Y_dv_var

    torch.manual_seed(765)
    model = logistic_regression.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()

    #train_model
    model_trained, losses, accuracies = logistic_regression.train_model(loss,model,
                                                       X_tr_var,
                                                       Y_tr_var,
                                                       X_dv_var=X_dv_var,
                                                       Y_dv_var = Y_dv_var,
                                                       num_its=800,
                                                       optim_args={'lr':1})
    
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert (acc >= 0.85)

def test_d7_3_competition_dev4():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-dev.preds.npy'),Y_dv_var.data.numpy())
    assert (acc >= 0.86)
"""
def test_d7_3_competition_test1():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-test.preds.npy'),Y_te_var.data.numpy())
    assert (acc >= 0.835)

def test_d7_3_competition_test2():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-test.preds.npy'),Y_te_var.data.numpy())
    assert (acc >= 0.84)

def test_d7_3_competition_test3():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-test.preds.npy'),Y_te_var.data.numpy())
    assert (acc >= 0.85)

def test_d7_3_competition_test4():
    global Y_dv_var
    acc = evaluation.acc(np.load('competition-test.preds.npy'),Y_te_var.data.numpy())
    assert (acc >= 0.86)
    """
