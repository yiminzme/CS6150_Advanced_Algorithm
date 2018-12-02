# -*- coding: utf-8 -*-
"""
Multiprocessing version

This program extract huge 128-dimensional feature vectors form a lot of images:
    1. Read each image file in the given directory
    2. Extract thounds of vectors from each image.
    3. Concatenate all the vectors.
    4. Save the all these 128-dimensional vectors into a file.
    
How to use it?
    1. Set the images_dir, the top directory of all the image files
    2. Set the output_file, the name of the file to store the result data.
    3. Set the image_file_limit, the limit number of the images to be processed.
        There are must more actual images than this limit.
    4. Run. After finish, you will see the total vectors, as well as the mean, 
        min, and max vectors of each image.

Libraries:
    cv2: opencv-contrib: https://anaconda.org/michael_wild/opencv-contrib
        Note: Other Opencv distributions don't come with SIFT, which is 
        necessary to extract feature vectors from images. 
        This distribution may also work, I have not tried.
            pip install opencv-contrib-python

Reference:    
    Where did SIFT and SURF go in OpenCV 3?
    https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
    
    Multiprocessing: use tqdm to display a progress bar
    https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
    
Created on Wed Nov 28 20:36:15 2018

@author: benwei
"""

import numpy as np
import os
import cv2 # must be compiled from opencv-contrib
from tqdm import tqdm # progress bar
import multiprocessing as mp
import argparse


# define a example function
def vectors_from_image(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    if descs is not None:
        return descs.astype(int)
    else:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            formatter_class = argparse.ArgumentDefaultsHelpFormatter,
            description = "Extract feature vectors from images.",
            epilog = '''A feature vector has 128 integers. \n
            The number of vectors of each usual image may be hundrads
            to thousands, depends on the complexity of the image. ''')
    parser.add_argument('-i', default = '../data/original/101_ObjectCategories/',
                        help='Input dir, the top dir of all the images to be process.')
    parser.add_argument('-o', default = '../data/Caltech101_small',
                        help="Output filename, ext '.npy' will be append automatictly.")
    parser.add_argument('-l', default = '10', 
                        help = "Number of images to be process. use 'all' for \
                        unlimit, process all images.")
    args = parser.parse_args()
    
    vectors = mp.Queue() # The matrix to collect all the vectors
    n_vectors = mp.Queue()
    image_files = []
    for root, dirs, files in os.walk(args.i):
        for file in files:
            image_files.append(os.path.join(root, file))
    
    if args.l != 'all':
        image_files = image_files[:int(args.l)]
    
    pool = mp.Pool()
    image_vectors = list(tqdm(pool.imap(vectors_from_image, image_files), total=len(image_files)))

    result = np.concatenate([x for x in image_vectors if x is not None], axis = 0)
    np.save(args.o, result)
    n_vectors = [len(m) for m in image_vectors if m is not None]
    print()
    print("total, mean, min, max: ", len(result), len(result)/len(n_vectors), min(n_vectors), max(n_vectors))