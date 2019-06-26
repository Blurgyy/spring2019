#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import sys
import numpy as np
from matplotlib import pyplot as plt

def plot(fname, ):
    fn_name = "plot";
    try:
        lines = None;
        xaxis = [];
        yaxis = [];
        with open(fname) as f:
            lines = f.readlines();
        for i in range(len(lines)):
            prec = float(lines[i].strip());
            xaxis.append(i+1);
            yaxis.append(prec);
        fig = plt.figure();
        y1 = fig.add_subplot(111);
        y1.plot(xaxis, yaxis, label = "precision");
        y1.axis([1, xaxis[-1], yaxis[0], 1.0]);
        y1.legend(loc = 1);
        ## define learning rate below
        x = np.arange(1, 101, 1);
        # y = np.ones(100); # constant
        # y = (100 - x) / (100); # linear
        # y = 1 / x; # hyperbola
        y = 1 / (1 + np.exp(x - 100 / 2)) # sigmoid
        # y = (np.arctan(-(x - 100/2)) + np.pi/2) / np.pi; # arctan
        y2 = y1.twinx();
        y2.plot(x, y, 'orange', label = "learning rate");
        y2.legend(loc = 2);
        plt.grid(True);
        plt.savefig(fname.split('.')[-2] + '.png');
        plt.show();
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

def main():
    fn_name = "main";
    try:
        fname = ".training_precision.log";
        if(len(sys.argv) == 2):
            fname = sys.argv[1];
        plot(fname);
    except Exception as e:
        print("%s(): %s" % (fn_name, e));

if(__name__ == "__main__"):
    main();
