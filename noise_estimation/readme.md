# Noise Estimation 🔊

In this repository we present several scripts that aim at quatifiying the level of noise on an image. This task is pretty difficult and has lead to lots 
of research studies. You will mainly find here the python implementations of some published research papers. 

## Boundary Discriminative Noise Detection : `bdnd.py`

This script implements the boundary discriminative noise detection (BDND) noise detection method which was proposed by Pei-Eng Ng and Kai-Kuang Ma in 2006 [[1]](#1).
The method is pretty straightforward to understand and very well explained in the [publication](https://www.researchgate.net/publication/281951781_Noise_Detection_in_Images_using_Moments). 
The current version of code lacks efficiency and has to be further improve later on. <br><br>

## Noise detection in images using Moments : `moments.py`

This methods was introduced by Maragatham et al. on 2015 [[2]](#2). It has the main advantage to remain quite easy to understand. The idea is to first
make a Discrete Cosine Transform of sub blocks of an image. Then we analyse the kurtosis of each frequency for all sub blocks. As we know that noise is 
more likely to be present at high frequencies, we then compute the Sum of Absolute Deviation of the kurtosis. <br><br>

# References 📋

<a id="1">[1]</a> Ng, Pei-Eng, and Kai-Kuang Ma ["A switching median filter with boundary discriminative noise detection for extremely corrupted images"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.105.6888&rep=rep1&type=pdf) </br>
IEEE transactions on image processing 15.6 (2006): 1506-1516.

<a id="2">[2]</a> Maragatham, G., S. Md Mansoor Roomi, and P. Vasuki ["Noise Detection in Images using Moments"](https://www.researchgate.net/publication/281951781_Noise_Detection_in_Images_using_Moments) </br>
Research Journal of Applied Sciences, Engineering and Technology 10.3 (2015): 307-314.

