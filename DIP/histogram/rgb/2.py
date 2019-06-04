#!/usr/bin/python
#-*- coding: utf-8 -*-
__author__ = "zgy";

import cv2 as cv
import sys
import math

def rgb2hsi(imIn):
    try:
        ret = imIn.copy();
        B, G, R = cv.split(imIn);
        B = B / 255;
        G = G / 255;
        R = R / 255;
        eps = 1e-8;
        PI = math.acos(-1.0);
        rows, cols = imIn.shape[0], imIn.shape[1];
        for i in range(rows):
            for j in range(cols):
                b = B[i, j];
                g = G[i, j];
                r = R[i, j];
                theta = 0;
                H = 0;
                S = 0;
                I = (r + g + b) / 3;
                if(I < eps):
                    S = 0;
                else:
                    S = 1 - (3 * min(r, g, b)) / (r + g + b);
                    if(r == g and g == b):
                        H = 0;
                    else:
                        theta = math.acos((0.5 * ((r-g) + (r-b))) / (((r-g) ** 2 + (r-b) * (g-b)) ** 0.5));
                        if(b <= g):
                            H = theta;
                        else:
                            H = 2 * PI - theta;
                    H /= 2 * PI;
                H *= 255;
                S *= 255;
                I *= 255;
                ret[i, j, 0] = H;
                ret[i, j, 1] = S;
                ret[i, j, 2] = I;
        return ret;
    except Exception as e:
        print("rgb2hsi(): %s" % (e));

def hsi2rgb(imIn):
    try:
        
    except Exception as e:
        print("hsi2rgb(): %s" % (e));

def main():
    try:
        if(len(sys.argv) != 2):
            raise Exception("invalid argument");
        imIn = cv.imread(sys.argv[1]);
        hsiImg = rgb2hsi(imIn);
        cv.imshow("test", hsiImg);
        cv.waitKey(0);
    except Exception as e:
        print("main(): %s" % (e));


if(__name__ == "__main__"):
    main();
