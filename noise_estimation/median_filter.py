
# ---------- IMPORTS ---------------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- MEDIAN FILTERING FUNCTION ---------------------------------------------------------------

def median_filter(image:np.array, kernel_size:int):
    
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

    for i in range(k, h+k):
        for j in range(k, w+k):
            px = image_[i, j]
            window = image_[i-k:i+k+1,j-k:j+k+1]
            v0 = np.sort(window.flatten())
            # print(window, "\n")
            # print("v0 : ", v0,"\n")
            med = np.median(v0)
            vD = np.abs(v0[:-1] - v0[1:])
            # print("vD : ", vD,"\n")
            
            b1 = v0[np.argmax(vD[:(l//2)+1])]
            b2 = v0[np.argmax(vD[(l//2)+2:])+(l//2)+2]
            # print("b1 : ", b1, "med : ", med, "b2 : ", b2)
            
            if b1 <= px <= b2: n+=1

    return(n/(h*w))

# ---------- MAIN ------------------------------------------------------------------------------------

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="path to image for noise estimation")
    args = parser.parse_args()

    image = plt.imread(args.image_path)

    median_filter(image, kernel_size=21)

    