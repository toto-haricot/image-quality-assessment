
# ---------- IMPORTS ---------------------------------------------------------------------------------

import argparse

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from time import time

# ---------- MEDIAN FILTERING FUNCTION ---------------------------------------------------------------

def count_noisy_pixels(image:np.array, kernel_size:int):
    
    assert len(image.shape) == 2, "image must be two-dimensionnal"
    assert kernel_size % 2 == 1, "kernel size must be impair"

    n = 0
    k = kernel_size//2
    l = kernel_size**2
    h, w = image.shape

    filtered = np.zeros((h,w))

    # zero padding 
    image_ = np.zeros((h+2*k, w+2*k))
    image_[k:-k, k:-k] = image

    pool = mp.Pool(mp.cpu_count())

    for i in range(k, h+k):
        for j in range(k, w+k):

            n += pool.apply(is_noisy, args=(image_, i, j, kernel_size))

            # px = image_[i, j]
            # window = image_[i-k:i+k+1,j-k:j+k+1]
            # v0 = np.sort(window.flatten())

            # med = np.median(v0)
            # vD = np.abs(v0[:-1] - v0[1:])
            
            # b1 = v0[np.argmax(vD[:(l//2)])]
            # b2 = v0[np.argmax(vD[(l//2):])+(l//2)]
            # print("b1 : ", b1, "med : ", med, "b2 : ", b2)
            
            # if b1 <= px <= b2: n+=1

    return(n/(h*w))

# ---------- IS_NOISY() FUNCTION ---------------------------------------------------------------------

def is_noisy(image:np.array, i:int, j:int, kernel_size:int):
    k = kernel_size//2
    l = kernel_size**2
    px = image[i,j]
    window = image[i-k:i+k+1,j-k:j+k+1]
    v0 = np.sort(window.flatten())
    med = np.median(v0)
    vD = np.abs(v0[:-1] - v0[1:])
            
    b1 = v0[np.argmax(vD[:(l//2)])]
    b2 = v0[np.argmax(vD[(l//2):])+(l//2)]

    if b1 <= px <= b2: return 1
    return 0

# ---------- MAIN ------------------------------------------------------------------------------------

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="path to image for noise estimation")
    args = parser.parse_args()

    image = plt.imread(args.image_path)[:,:,0]

    t1 = time()

    count_noisy_pixels(image, kernel_size=21)

    t2 = time()

    print(f"Execution time : {t2-t1} seconds")
    