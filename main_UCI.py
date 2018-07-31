import tensorflow as tf
import numpy as np
import os
import glob
import time
import argparse
import cProfile
import json
import sys
import shlex
import subprocess
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.io
import h5py
import csv
import os
import scipy.misc
import scipy.io
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon, ranksums
from argparse import ArgumentParser
import logging
import sys
import copy
import numpy as np
import os
import parse_image_seg_UCI

import model_generator
import argparse
import csv

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("--conic", type=str, default='Rectangular', help="if conic")
    parser.add_argument("--bn", type=int, default=0, help="batch normalization")
    parser.add_argument("--keep", type=float, default=0.5, help="keep_prob for dropout")
    parser.add_argument("--dataset_index", type=int, default=47, help="index_of_dataset")
    parser.add_argument("--batchsize", type=int, default=20, help="batchsize")
    parser.add_argument("--epochs", type=int, default=500, help="num_of_epochs")
    parser.add_argument("--sample_size", type=int, default=5, help="num_of_samples_for_var_est")
    parser.add_argument("--eps", type=float, default=0.1, help="epsilon_for_stability")
    parser.add_argument("--use_reg", type=int, default=0, help="use regularizer")
    parser.add_argument("--depth", type=int, default=4, help="net depth")
    parser.add_argument("--n_start", type=int, default=256, help="layer_width")
    parser.add_argument("--lr", type=float, default=0.01, help="learning_rate")
    parser.add_argument("--gamma", type=float, default=0.01, help="gamma_for_reg")
    parser.add_argument("--activation", type=str, default='relu', help="activation_type")
    parser.add_argument("--data_path", type=str, default='/Users/etailittwin/Apple/data', help="path_of_all_datasets")
#    parser.add_argument("--data_path", type=str, default='/private/home/wolf/etaibn/data', help="path_of_all_datasets")
    args = parser.parse_args()

    return args


args = parseArguments()

use_reg = args.use_reg
batchsize = args.batchsize
depth = args.depth
architecture = 'rectangle'
n_start = args.n_start
lr = args.lr
if args.conic=='Rectangular':
    conic = 0
else:
    conic = 1
epochs = args.epochs
if args.activation=='elu':
    activation = tf.nn.elu
if args.activation=='relu':
    activation = tf.nn.relu
if args.activation=='selu':
    activation = tf.nn.selu
if args.activation=='tanh':
    activation = tf.nn.tanh
bn = args.bn
keep = args.keep
data_path = args.data_path

FILENAME_DATA = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/image-segmentation_py.dat'
FILENAME_LABELS = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/labels_py.dat'
# FILENAME_TEST = r'datasets/image-segmentation/segmentation.test'
FILENAME_INDEXES_TEST = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/folds_py.dat'
FILENAME_VALIDATION_INDEXES = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/validation_folds_py.dat'
assert_values_flag = True
dataset_dict = {'name': 'image_segmentation', 'file_names': (FILENAME_DATA, FILENAME_LABELS, FILENAME_INDEXES_TEST, FILENAME_VALIDATION_INDEXES),
                'assert_values_flag': assert_values_flag,
                'validation_train_ratio': 5.0,
                'test_alldata_ratio': 300.0 / 330}
trainset = parse_image_seg_UCI.Dataset(dataset_dict)

# datasets = os.listdir(data_path)
# dataset_idx = args.dataset_index
# trainset = UCIDatasetGeneral(dataset = datasets[dataset_idx], root=data_path, train=True)

# num_of_labels = trainset.num_classes
num_of_labels = trainset.get_num_of_labels()
# input_dim = trainset.input_dim()
input_dim = trainset.get_dimensions()[0]

data_node  = tf.placeholder(tf.float32, shape=[None, input_dim], name='data')
labels_node = tf.placeholder(tf.int32, shape=[None], name='labels')
lr_node = tf.placeholder(tf.float32,shape=(), name='learning_rate') 
gm_node = tf.placeholder(tf.float32,shape=(), name='gamma') 
isTrain_node = tf.Variable(False, name='istrainvar', trainable = False)
tf.add_to_collection('istrainvar',isTrain_node)

train_data = trainset.get_train_set()
val_data = trainset.get_validation_set()
test_data = trainset.get_test_set()

train_labels = trainset.get_train_labels()
val_labels = trainset.get_validation_labels()
test_labels = trainset.get_test_labels()

if args.activation=='selu':
    logits = model_generator.conic_architecture_selu(data_node, args.n_start, num_of_labels, activation, args.depth, args.sample_size, args.eps, isTrain_node, keep, bn, conic)
else:
    logits = model_generator.conic_architecture(data_node, args.n_start, num_of_labels, activation, args.depth, args.sample_size, args.eps, isTrain_node, keep, bn, conic)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = labels_node))

l2_loss = tf.get_collection('l2_loss')
l2_loss = tf.add_n(l2_loss)

l2_norm = tf.get_collection('l2_norm')
l2_norm = tf.add_n(l2_norm)

if use_reg==1:
    loss = loss + 0.0000*l2_loss + gm_node*l2_norm
else:
    loss = loss + 0.0000*l2_loss + 0*l2_norm

t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
optimizer = tf.train.MomentumOptimizer(lr_node, 0.9)
##optimizer = tf.train.AdamOptimizer(lr_node)
grads_and_vars = optimizer.compute_gradients(loss, var_list = t_vars)
clip_constant = 1
grads_and_vars_rescaled = [(tf.clip_by_norm(gv[0],clip_constant),gv[1]) for gv in grads_and_vars]
train_op_net = optimizer.apply_gradients(grads_and_vars_rescaled)
        
#train_op_net = optimizer.minimize(loss = loss, var_list = t_vars)
#grads_and_vars = optimizer.compute_gradients(loss, var_list = t_vars)
#grads_and_vars_rescaled = [(tf.clip_by_norm(gv[0],0.1),gv[1]) for gv in grads_and_vars]
#train_op_net = optimizer.apply_gradients(grads_and_vars_rescaled)


predictions = logits
correct_prediction = tf.equal(tf.cast(tf.argmax(predictions, 1), tf.int32), labels_node)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
err = 1-tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#err = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()
feed_dict = {}
feed_dict[lr_node] = lr
ini = tf.initialize_all_variables()
session.run(ini)         
error_plot_train = []
error_plot_test = []
error_plot_val = []
min_error_val = 1
min_error_test = 1
test_at_val = 1
feed_dict[gm_node] = args.gamma
test_at_min_val_ma = 1
val_ma = []
for i in range(epochs):
#    if i>100:
#        feed_dict[gm_node] = args.gamma
#        feed_dict[lr_node] = 0.005
    if i>200:
        feed_dict[gm_node] = args.gamma
        feed_dict[lr_node] = 0.001
    avg_error = 0
    perm = np.random.permutation(train_data.shape[0])
    perm = np.concatenate((perm,perm),0)
    itter = 0
    session.run(isTrain_node.assign(True))
    for idx in range(0,train_data.shape[0],batchsize):
        batch_idx = perm[idx:idx + batchsize]
        feed_dict[data_node] = train_data[batch_idx,:]
        feed_dict[labels_node] = train_labels[batch_idx]
        _, loss_val, error_val = session.run([train_op_net, loss, err], feed_dict = feed_dict)
        avg_error+=error_val
        itter+=1
    error_plot_train.append(avg_error/(itter))
    print(i)
    
    
    session.run(isTrain_node.assign(False))
    
    avg_test_error = 0
    for idx in range(test_data.shape[0]):
        feed_dict[data_node] = test_data[idx:idx+1,:]
        feed_dict[labels_node] = test_labels[idx:idx+1]
        error_test = session.run(err, feed_dict = feed_dict)
        avg_test_error+=error_test
    error_plot_test.append(avg_test_error/(test_data.shape[0] + 1))
    if avg_test_error/(test_data.shape[0] + 1)<min_error_test:
        min_error_test = avg_test_error/(test_data.shape[0] + 1)
        
    avg_val_error = 0
    for idx in range(val_data.shape[0]):
        feed_dict[data_node] = val_data[idx:idx+1,:]
        feed_dict[labels_node] = val_labels[idx:idx+1]
        error_val = session.run(err, feed_dict = feed_dict)
        avg_val_error+=error_val
    error_plot_val.append(avg_val_error/(val_data.shape[0] + 1))
    if avg_val_error/(val_data.shape[0] + 1)<min_error_val:
        min_error_val = avg_val_error/(val_data.shape[0] + 1)
        test_at_val = error_plot_test[-1]
    if i>10:
        val_ma.append(np.mean(np.asarray(error_plot_val[-10:])))
        if min(val_ma) == val_ma[-1]:
            test_at_min_val_ma = error_plot_test[-1]
#    plt.figure(1)
#    plt.plot(error_plot_test,'b',error_plot_val,'r')
#    plt.pause(0.005)
    print('_______________________________________________________________')
    print("activation: %s, n_start: [%5d], dataset_index: [%5d], eps: %.8f, use_reg: [%5d], activation: %s, gamma: %.8f, bn: [%5d], keep: %.8f, conic: [%5d], depth: [%5d]"  % (str(0), args.n_start, args.dataset_index, args.eps, args.use_reg, str(args.activation), args.gamma, bn, keep, conic, depth))
    print("Testing: epoch [%5d], min_val error: %.8f, min_test error: %.8f" % (i, min_error_val, min_error_test))
    print("Testing: epoch [%5d], val error: %.8f, test error: %.8f, train error: %.8f, test at_min_val_error: %.8f, test at_min_ma_val_error: %.8f" % (i, error_plot_val[i], error_plot_test[i], error_plot_train[i], test_at_val, test_at_min_val_ma))
    
