import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import copy
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.autograd import Variable
import argparse


datapath = "../data/"
df = pd.read_csv("view_list.csv", names=['datasets'],sep='	')
df_train, df_test = {}, {}
max_phi, min_phi = 180, -180
max_theta, min_theta = 90, -90
# seq_length, delay_length = 30, 30


def load_dataset_phi(datapath, dataset, dataset_test,seq_length, delay_length):
    x_all, y_all = [], []
    #train data
    df_train = pd.read_csv("{}{}.txt".format(datapath,df.iloc[dataset][0]), sep='\t', names=['phi', 'theta'])
    training_data = df_train.iloc[:,0:1].values
    training_data = transform(training_data, min_phi, max_phi)
    x, y = sliding_windows(training_data, seq_length, delay_length)
    x_all.append(x)
    y_all.append(y)
    x_train = np.concatenate(x_all,0)
    y_train = np.concatenate(y_all,0)
    # test data
    
    df_test = pd.read_csv("{}{}.txt".format(datapath,df.iloc[dataset_test][0]), sep='\t', names=['phi', 'theta'])
    test_set = df_test.iloc[:,0:1].values
    test_data = transform(test_set, min_phi, max_phi)
    x_test, y_test = sliding_windows(test_data, seq_length, delay_length)
    return x_train, y_train, x_test, y_test

def load_dataset_theta(datapath, dataset, dataset_test, seq_length, delay_length):
    x_all, y_all = [], []
    #train data
    df_train = pd.read_csv("{}{}.txt".format(datapath,df.iloc[dataset][0]), sep='\t', names=['phi', 'theta'])
    training_data = df_train.iloc[:,1:2].values
    training_data = transform(training_data, min_phi, max_phi)
    x, y = sliding_windows(training_data, seq_length, delay_length)
    x_all.append(x)
    y_all.append(y)
    x_train = np.concatenate(x_all,0)
    y_train = np.concatenate(y_all,0)
    # test data
    
    df_test = pd.read_csv("{}{}.txt".format(datapath,df.iloc[dataset_test][0]), sep='\t', names=['phi', 'theta'])
    test_set = df_test.iloc[:,0:1].values
    test_data = transform(test_set, min_phi, max_phi)
    x_test, y_test = sliding_windows(test_data, seq_length, delay_length)
    return x_train, y_train, x_test, y_test

def load_dataset_full(datapath, dataset, dataset_test, seq_length, delay_length):
    x_train, y_train, x_test, y_test = load_dataset_phi(datapath, dataset, dataset_test)
    x_train_theta, y_train_theta, x_test_theta, y_test_theta = load_dataset_theta(datapath, dataset, dataset_test)
    return x_train, y_train, x_test, y_test, x_train_theta, y_train_theta, x_test_theta, y_test_theta

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help="number of rounds of training")
    parser.add_argument('--hidden_size', type=int, default=2,
                        help="RNN's hidden size")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="RNN's number of layers")
    parser.add_argument('--seq_length', type=int, default=1,
                        help="History window size")
    parser.add_argument('--num_run', type=int, default=10,
                        help="Number of runs")

    args = parser.parse_args()
    return args

def rmse(pred, gt):
    return np.sqrt(((pred - gt) ** 2).mean())

def transform(X, min_X, max_X):
    return (X - min_X)/(max_X - min_X)
def inverse_transform(X, min_X, max_X):
    return X * (max_X - min_X) + min_X
def sliding_windows(data, seq_length, delay_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-delay_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length + delay_length - 1]

        ## calibrate data at boudaries ##
        flg = True
        if flg:
            x.append(_x)
            y.append(_y)

    return np.array(x),np.array(y)