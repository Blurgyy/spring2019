#!/usr/bin/python
#-*- coding: utf-8 -*-
__author__ = "zgy";

import sys
import cv2 as cv
# import numpy as np


def histogram_eq(imIn, GRAYSCALE_range = 256):
    try:
        ret = imIn.copy();
        mp = [];
        for i in range(GRAYSCALE_range):
            mp.append(0);
        rows, cols = int(imIn.shape[0]), int(imIn.shape[1]);
        for i in range(rows):
            for j in range(cols):
                mp[imIn[i, j]] += 1;
        for i in range(1, GRAYSCALE_range):
            mp[i] += mp[i-1];
        for i in range(GRAYSCALE_range):
            mp[i] = 1.0 * (GRAYSCALE_range -1) * mp[i] / mp[GRAYSCALE_range-1] + 0.5;
            mp[i] = int(mp[i]);
        for i in range(rows):
            for j in range(cols):
                ret[i, j] = mp[ret[i, j]];
        return ret;
    except Exception as e:
        print("histogram_eq(): %s" % e);


def main():
    try:
        if(len(sys.argv) != 2):
            raise Exception("invalid argument");
        img = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE);

        y = histogram_eq(img);
        cv.imshow("test", img);
        cv.imshow("after", y);
        cv.waitKey(0);
    except Exception as e:
        print("main(): %s" % e);


if(__name__ == '__main__'):
    try:
        main();
    except Exception as e:
        print("global: %s" % e);
