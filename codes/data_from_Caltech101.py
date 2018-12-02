# -*- coding: utf-8 -*-
"""
Don't run it, use multiprocessing version instead.
    But the code is easy to understand.
    
The output of the last run:
    100%|██████████| 9145/9145 [1:33:56<00:00,  1.06s/it]
    Total vectors:  4183192
The resulting file has size 1.99 GB.

Created on Wed Nov 28 20:36:15 2018

@author: benwei
"""

import numpy as np
import os
import cv2
from tqdm import tqdm # progress bar

images_dir = "./data/original/101_ObjectCategories/"
output_file = "./data/Caltech101_full"

vectors = np.empty([0, 128], dtype=int) # The matrix to collect all the vectors
n_vectors = []

n_files = sum([len(files) for r, d, files in os.walk(images_dir)])
pbar = tqdm(total=n_files)
sift = cv2.xfeatures2d.SIFT_create()
for root, dirs, files in os.walk(images_dir):
    for file in files:
        pbar.update(1)
        image = cv2.imread(os.path.join(root, file))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, descs) = sift.detectAndCompute(gray, None)
        if descs is not None:
            n_vectors.append(len(descs))
            vectors = np.concatenate((vectors,descs.astype(int)), axis = 0)
        else:
            n_vectors.append(0)

pbar.close()
np.save(output_file,vectors)
print("Total vectors: ", len(vectors))