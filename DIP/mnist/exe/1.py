#!/usr/bin/python3
# -*- coding: utf-8 -*-
__author__ = "zgy";

import os 
import pickle 
import numpy as np 
from PIL import Image 
# import torch

traning_path = "dat/training";
traning_img = [];

print("reading pickle file..")
with open("traning_set.pickle", 'rb') as f:
    traning_img = pickle.load(f);
print("pickle file read in memory");


# cnt = 0;
# for root, dirs, files in os.walk(traning_path):
#     for file in files:
#         if(file.split('.')[-1] == "png"):
#             img = Image.open(root + '/' + file);
#             arr = np.array(img);
#             traning_img.append(arr.reshape(-1, 1));
#             cnt += 1;
#             if(cnt % 1000 == 0):
#                 print(cnt);
# print("traning_img in memory");

# print("dumping..");
# with open("traning_set.pickle", 'wb') as f:
#     pickle.dump(traning_img, f);

