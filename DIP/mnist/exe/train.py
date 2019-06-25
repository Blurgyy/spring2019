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
        if("--retrain" in sys.argv):
            retrain = True;
        if(os.path.exists(w_dmp_fname) and not retrain):
            try:
                print("loading weight...", end = '');
                with open(w_dmp_fname, 'rb') as f:
                    w = pickle.load(f);
                print("Done")
            except Exception as e:
                print("read from file failed: %s" % e);
                w = np.zeros(7840).reshape(10, 784);
                print("weight reinitialized");
        else:
            w = np.zeros(7840).reshape(10, 784);
            print("weight reinitialized");
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def normalize(x, Range = 1, ):
    fn_name = "normalize";
    try:
        eps = 1e-8;
        ret = None;
        if(np.sum(x) < eps):
            ret = np.ones(x.shape) * Range;
        else:
            ret = x / np.sum(x) * Range;
        return ret;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def backup_weights(w, ):
    fn_name = "backup_weights";
    try:
        w_dmp_fname = "../dmp/w.pickle";
        with open(w_dmp_fname, 'wb') as f:
            pickle.dump(w, f);
        return True;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return False;

def update_weights(w, x, score, learning_rate, Lambda, backup = False):
    fn_name = "update_weights";
    try:
        X = x[0].reshape(1, -1);
        gt = x[1];
        # prob = normalize(score, 10);
        prob = score;
        prob -= np.max(prob);
        prob = np.exp(prob) / np.sum(np.exp(prob));
        dL_dw = np.dot(prob, X);
        dL_dw[gt] = dL_dw[gt] - X;
        # w = w - learning_rate * dL_dw; # adding this and the program will not run correctly
        w -= learning_rate * dL_dw; # while this runs perfectly, idk why

        dLr = X * 2;
        w -= Lambda * dLr;
        # print('\n', w.min(), w.max());
        # input();
        if(backup):
            backup_weights(w);
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
            
            idx = np.argmax(score);
            # print("predicted:\t%d\nground truth:\t%d" % (idx, training_imgs[i][1]));
            if(idx == training_imgs[i][1]):
                YES += 1;
            else:
                NO += 1;
            # print(YES, NO);
            ratio = 100 * YES/(YES+NO);
            # print("\rtrained \033[1;37m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%" % (YES + NO, YES, NO, ratio), end = '     ');
            print("\rtrained \033[1;34m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%, (%g, %g)" % (YES + NO, YES, NO, ratio, np.amax(score), np.amin(score)), end = '     ');
            update_weights(w, training_imgs[i], score, learning_rate, i % 1000 == 999);
            # for i in range(score.shape[0]):
            #     print(score[i][0]);
            # input();
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        w = init_weights();
        epoch = 100;
        if("--epoch" in sys.argv):
            try:
                epoch = int(sys.argv[sys.argv.index("--epoch")+1]);
            except Exception as e:
                pass;
        print("\033[1;37mstarting training\033[0m: epoch = %d" % (epoch));
        for i in range(epoch):
            print("\ntraining epoch %d/%d:" % (i+1, epoch));
            training_imgs = load_training_set();
            Lambda = (epoch - i) / (10 * epoch);
            learning_rate = (epoch - i) / (epoch);
            # Lambda = 0.01 / (i + 1);
            # learning_rate = 1 / (i + 1);
            train(training_imgs, w, Lambda, learning_rate);
            backup_weights(w);
            print();
    except Exception as e:
        print("%s(): %s" % (fn_name, e));


if(__name__ == "__main__"):
    main();
