#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import os 
import pickle 
import numpy as np 
from PIL import Image 
# import torch
# from torch.autograd import Variable 

training_set_dmp_fname = "../dmp/training_set.pickle";
# testing_set_dmp_fname = "../dmp/testing_set.pickle";
training_imgs = [];
# testing_img = [];

with open(training_set_dmp_fname, 'rb') as f:
    training_imgs = pickle.load(f);
print("training set loaded")

w = np.zeros(7840).reshape(10, 784);
print(np.log(np.exp(1)));

def regularization_loss(w, ):
    try:
        rows, cols = w.shape;
        ret = 0;
        for i in range(rows):
            for j in range(cols):
                ret += w[i][j] ** 2;
        return ret;
    except Exception as e:
        print("regularization_loss(): %s" % e);
        return None;

def softmax_loss(score, gt, ):
    try:
        ret = 0;
        smax = score;
        rows, cols = smax.shape;
        sum = 0;
        for i in cols:
            smax[0][i] = np.exp(score[0][i]);
            sum += smax[0][i];
        ret = -np.log(smax[0][gt]/sum);
        return ret;
    except Exception as e:
        print("softmax_loss(): %s" % e);
        return None;

def loss(w, score, ground_truth):
    try:
        return softmax_loss(score, ground_truth) + regularization_loss(w);
    except Exception as e:
        print("loss(): %s" % e);
        return None;

def train(training_imgs, w, ):
    try:
        size = len(training_imgs);
        for i in range(size):
            score = np.dot(w, training_imgs[i][0]);
            print(score, training_imgs[i][1]);
        return w;
    except Exception as e:
        print("train(): %s" % e);

train(training_imgs, w);
