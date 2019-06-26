#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import sys
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
        plt.plot(xaxis, yaxis);
        plt.axis([1, xaxis[-1], yaxis[0], 1.0]);
        plt.savefig('training_precision.png');
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
