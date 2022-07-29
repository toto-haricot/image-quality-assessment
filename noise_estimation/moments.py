"""Python implementation of paper Noise Detection in Images using Moments
"""

# ---------- IMPORTS ---------------------------------------------------------------------------------

import os
import cv2
import time
import tqdm
import argparse
import scipy.stats

import numpy as np
import pandas as pd

# ---------- ARGUMENTS PARSING -----------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", type=str, required=False, help="path to image")
parser.add_argument("--folder_path", type=str, required=False, help="path to images folder")

args = parser.parse_args()

image_path = args.image_path
folder_path = args.folder_path

# ---------- FUNCTIONS -------------------------------------------------------------------------------

def dct(image:np.array, block_size:int):

    c = 0
    h, w = image.shape
    n_h, n_w = h//block_size, w//block_size
    
    dct_ = np.zeros((n_h*n_w, block_size**2))

    for i in range(n_h):
        for j in range(n_w):

            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_[c] = cv2.dct(np.float32(block)).flatten()
            c+=1

    return dct_


def kurtosis(X:np.array):

    return np.apply_along_axis(scipy.stats.kurtosis, 0, X)


def sad(kurtosis_serie:list):

    n = len(kurtosis_serie)
    Km = np.mean(kurtosis_serie)
    return(np.sum([abs(k-Km) for i,k in enumerate(kurtosis_serie) if i > 0.625*n]))

# ---------- MAIN ------------------------------------------------------------------------------------    

if __name__ == '__main__': 

    t1 = time.time()

    if image_path: 

        image = cv2.imread(image_path)[:,:,0]
        image_sad = sad(kurtosis(dct(image, block_size=16)))

        t = time.time() - t1

        print(f'Image kurtosis sum of absolute deviation : {image_sad}')
        print(f'Time to process : {round(t, 4)} seconds')

    if folder_path: 

        df = pd.DataFrame(columns=["image_name", "noise_estimation"])

        images = os.listdir(folder_path)

        for image_name in tqdm.tqdm(images):

            # print(f'Image : {image_name}')

            if not image_name.endswith('.jpg'): continue

            image_path = os.path.join(folder_path, image_name)

            image = cv2.imread(image_path)[:,:,0]
            image_sad = sad(kurtosis(dct(image, block_size=16)))

            # print(f'kurtosis sum of absolute deviation : {image_sad}')

            df_ = pd.DataFrame({"image_name":[image_name], "noise_estimation":[image_sad]})
            df = pd.concat([df, df_], ignore_index=True, axis=0)

        t = time.time() - t1
        print(f'Time to process all folder : {round(t, 2)} seconds')

        df.to_csv("noise_estimation.csv")




