#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import os 
import sys
import pickle 
import numpy as np 
from PIL import Image 
import random 
# import torch
# from torch.autograd import Variable 


def load_training_set():
    fn_name = "load_training_set";
    try:
        training_set_dmp_fname = "../dmp/training_set.pickle";
        training_imgs = None;
        with open(training_set_dmp_fname, 'rb') as f:
            training_imgs = pickle.load(f);
            random.shuffle(training_imgs);
        print("training set loaded and shuffled");
        return training_imgs;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def init_weights():
    fn_name = "init_weights";
    try:
        w_dmp_fname = "../dmp/w.pickle";
        retrain = False;
        if(len(sys.argv) > 1 and sys.argv[1] == "--retrain"):
            retrain = True;
        if(os.path.exists(w_dmp_fname) and not retrain):
            try:
                with open(w_dmp_fname, 'rb') as f:
                    w = pickle.load(f);
            except Exception as e:
                print("read from file failed: %s" % e);
                w = np.zeros(7840).reshape(10, 784);
                print("weight reinitialized");
        else:
            w = np.zeros(7840).reshape(10, 784);
        return w;
    except Exception as E:
        print("%s(): %s" % (fn_name, e));
        return None;

def regularization_loss(w, ):
    fn_name = "regularization_loss";
    try:
        rows, cols = w.shape;
        ret = 0;
        for i in range(rows):
            for j in range(cols):
                ret += w[i][j] ** 2;
        return ret;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def softmax_loss(score, gt, ):
    fn_name = "softmax_loss";
    try:
        ret = 0;
        softmax = score;
        softmax -= np.max(softmax);
        softmax = np.exp(softmax) / np.sum(np.exp(softmax));
        crs_entropy_loss = -np.log(softmax);
        ret = crs_entropy_loss[gt][0];
        # print(ret);
        # input();
        return ret;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def total_loss(w, score, ground_truth, Lambda, ):
    fn_name = "total_loss";
    try:
        sloss = softmax_loss(score, ground_truth);
        rloss = regularization_loss(w);
        # print(sloss, rloss);
        return sloss + Lambda * rloss;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def update_weights(w, x, loss, learning_rate, backup = False):
    fn_name = "update_weights";
    try:
        grad = x[0];
        ground_truth = x[1];
        rows, cols = w.shape;
        for i in range(rows):
            for j in range(cols):
                # w[i][j] += learning_rate * loss * (-x[j][0]);
                if(i == ground_truth):
                    w[i][j] += learning_rate * loss * (grad[j][0]);
                else:
                    w[i][j] += learning_rate * loss * (-grad[j][0]);
        if(backup):
            w_dmp_fname = "../dmp/w.pickle";
            with open(w_dmp_fname, 'wb') as f:
                pickle.dump(w, f);
            # print("backup complete");
        # print(np.amax(w), np.amin(w), end = "\r");
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def train(training_imgs, w, Lambda, learning_rate, ):
    fn_name = "train";
    try:
        YES = 0;
        NO = 0;
        # random.shuffle(training_imgs);
        size = len(training_imgs);
        for i in range(size):
            score = np.dot(w, training_imgs[i][0]);
            loss = total_loss(w, score, training_imgs[i][1], Lambda);
            
            maxx = -1;
            idx = -1;
            for j in range(score.shape[0]):
                if(maxx < score[j][0]):
                    maxx = score[j][0];
                    idx = j;
            # print("predicted:\t%d\nground truth:\t%d" % (idx, training_imgs[i][1]));
            if(idx == training_imgs[i][1]):
                YES += 1;
            else:
                NO += 1;
            # print(YES, NO);
            ratio = 100 * YES/(YES+NO);
            print("trained \033[1;37m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m]) pics, precision: %.2f%%" % (YES + NO, YES, NO, ratio), end = '\r');
            # print("trained \033[1;37m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m]) pics, precision: %.2f%%, (%g, %g)" % (YES + NO, YES, NO, ratio, np.amax(w), np.amin(w)), end = '\r');
            update_weights(w, training_imgs[i], loss, learning_rate, i % 1000 == 999);
            # for i in range(score.shape[0]):
            #     print(score[i][0]);
            # input();
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        training_imgs = load_training_set();
        w = init_weights();
        Lambda = 0.02;
        learning_rate = 0.00000005;
        train(training_imgs, w, Lambda, learning_rate);
    except Exception as e:
        print("%s(): %s" % (fn_name, e));


if(__name__ == "__main__"):
    main();
