#!/usr/bin/python
# -*- coding: utf-8 -*- 
__author__ = "Blurgy";

import numpy as np
import pickle
import random

def load_test_set():
    fn_name = "load_test_set";
    try:
        testing_set_dmp_fname = "../dmp/testing_set.pickle";
        testing_imgs = None;
        with open(testing_set_dmp_fname, 'rb') as f:
            testing_imgs = pickle.load(f);
        random.shuffle(testing_imgs);
        return testing_imgs;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def load_weights():
    fn_name = "load_weights";
    try:
        w_dmp_fname = "../dmp/w.pickle";
        w = None;
        with open(w_dmp_fname, 'rb') as f:
            w = pickle.load(f);
        return w;
    except Exception as e:
        print("%s(): %s" % (fn_name, e));
        return None;

def test(testing_imgs, w, ):
    fn_name = "test";
    try:
        size = len(testing_imgs);
        YES = 0;
        NO = 0;
        for i in range(size):
            x = testing_imgs[i][0];
            gt = testing_imgs[i][1];
            score = np.dot(w, x);
            idx = np.argmax(score);
            if(idx == gt):
                YES += 1;
            else:
                NO += 1;
            ratio = YES / (YES + NO) * 100;
            print("\rvalidate: \033[1;37m%d\033[0m(\033[1;32m%d\033[0m/\033[1;31m%d\033[0m) pics, precision: %.2f%%" % (YES + NO, YES, NO, ratio), end = '     ');
        print();
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        testing_imgs = load_test_set();
        w = load_weights();
        test(testing_imgs, w);
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

if(__name__ == "__main__"):
    main();
