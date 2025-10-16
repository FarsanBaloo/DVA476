# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 17:14:27 2020

@author: rhr01
"""

# -*- coding: utf-8 -*-

# Please keep in mind that this is just startup code. You have to complete step 1 and step 2
#using CNN

import os
import numpy as np # linear algebra
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

#Data Handling

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob,string
#you have to change the path
path = 'D:/Projects/12_FutureE/Project solution/coil-100/*.png'
#list files
files=glob.glob(path)


import codecs
from tqdm import tqdm
def contructDataframe(file_list):
    """
    this function builds a data frame which contains 
    the path to image and the tag/object name using the prefix of the image name
    """
    data=[]
    for file in tqdm(file_list):
        data.append((file,file.split("/")[-1].split("__")[0]))
    return pd.DataFrame(data,columns=['path','label'])

df=contructDataframe(files)
#Then we have all the images paths whit its label in a dataframe.

#You can paste this line in console pannel to see all images in paths
#df.tail(10)

# Now you have to do the followings:
#step 1: you have to write codes for CNN classification
# Step 2: then you have to compare the results between ANN and CNN
print(' Wish you good luck');