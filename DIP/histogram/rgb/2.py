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
        B = B / 255.0;
        G = G / 255.0;
        R = R / 255.0;
        eps = 1e-8;
        pi = math.acos(-1.0);
        rows, cols = imIn.shape[0], imIn.shape[1];
        for i in range(rows):
            for j in range(cols):
                b = B[i, j];
                g = G[i, j];
                r = R[i, j];
                theta = 0;
                H = 0;
                S = 0;
                I = (r + g + b) / 3.0;
                if(I < eps):
                    S = 0;
                else:
                    S = 1 - (3 * min(r, g, b)) / (r + g + b);
                    if(r == g and g == b):
                        H = 0;
                    else:
                        theta = math.acos((0.5 * ((r-g) + (r-b))) / math.sqrt(((r-g) ** 2 + (r-b) * (g-b))));
                        if(b <= g):
                            H = theta;
                        else:
                            H = 2 * pi - theta;
                H /= 2 * pi;
                H *= 255;
                S *= 255;
                I *= 255;
                assert(H >= 0 and H <= 255);
                assert(S >= 0 and S <= 255);
                assert(I >= 0 and I <= 255);
                ret[i, j, 0] = H;
                ret[i, j, 1] = S;
                ret[i, j, 2] = I;
        return ret;
    except Exception as e:
        print("rgb2hsi(): %s" % (e));

def hsi2rgb(imIn):
    try:
        ret = imIn.copy();
        rows, cols = ret.shape[0], ret.shape[1];
        H, S, I = cv.split(imIn);
        H = H / 255.0;
        S = S / 255.0;
        I = I / 255.0;
        eps = 1e-8;
        pi = math.acos(-1.0);
        for i in range(rows):
            for j in range(cols):
                B = 0;
                G = 0;
                R = 0;
                if S[i, j] < eps:
                    R = I[i, j];
                    G = I[i, j];
                    B = I[i, j];
                else:
                    H[i, j] *= 360;
                    if(0 <= H[i, j] and H[i, j] <= 120):
                        H[i, j] = H[i, j] / 180 * pi;
                        R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j])) / (math.cos(pi/3 - H[i, j])));
                        B = I[i, j] * (1 - S[i, j]);
                        G = 3 * I[i, j] - (R + B);
                    elif(120 < H[i, j] and H[i, j] <= 240):
                        H[i, j] -= 120;
                        H[i, j] = H[i, j] / 180 * pi;
                        G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j])) / (math.cos(pi/3 - H[i, j])));
                        R = I[i, j] * (1 - S[i, j]);
                        B = 3 * I[i, j] - (R + G);
                    elif(240 < H[i, j] and H[i, j] <= 360):
                        H[i, j] -= 240;
                        H[i, j] = H[i, j] / 180 * pi;
                        B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j])) / (math.cos(pi/3 - H[i, j])));
                        G = I[i, j] * (1 - S[i, j]);
                        R = 3 * I[i, j] - (G + B);
                    else:
                        raise Exception("H =", H[i, j]);
                if(B > 1):
                    B = 1;
                if(G > 1):
                    G = 1;
                if(R > 1):
                    R = 1;
                B = int(B * 255);
                G = int(G * 255);
                R = int(R * 255);
                # print(B, G, R);
                assert(B >= 0 and B <= 255);
                assert(G >= 0 and G <= 255);
                assert(R >= 0 and R <= 255);
                ret[i, j, 0] = B;
                ret[i, j, 1] = G;
                ret[i, j, 2] = R;
        return ret;
    except Exception as e:
        print("hsi2rgb(): %s" % (e));

def histogram_eq(imIn, GRAYSCALE_range = 256):
    try:
        ret = imIn.copy();
        mp = [];
        for i in range(GRAYSCALE_range):
            mp.append(0);
        rows, cols = imIn.shape[0], imIn.shape[1];
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
        if(len(sys.argv) != 2 and len(sys.argv) != 3):
            raise Exception("invalid argument");
        imIn = cv.imread(sys.argv[1]);
        hsiImg = rgb2hsi(imIn);
        hsiImg[:, :, 2] = histogram_eq(hsiImg[:, :, 2]);
        imOut = hsi2rgb(hsiImg);
        if(len(sys.argv) == 3):
            cv.imwrite(sys.argv[2], imOut);
        cv.imshow("original", imIn);
        cv.imshow("test", imOut);
        cv.waitKey(0);
    except Exception as e:
        print("main(): %s" % (e));


if(__name__ == "__main__"):
    main();
