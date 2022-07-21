# IQA : Image Quality Assessment 🧐

This repository presents several model to quantify the quality of an image. An image quality can be assessed with comparison to its ground-truth, in
this case we talk about **full-reference assessement**. Another, more challenging, task consists in assessing an image quality without any reference.
The model should be able to extract some relevant features from the information to quantify image quality without any other information. In this case
we talk about **blind assessement**. 

## Full-Reference assessment

- 📝 **psnr.py** : The Peak Signal-to-Noise Ratio is a score between zero and one, expressed in decibels, that measures the proximity between a
ground truth image and its restored version. The higher quality the image has, the closer to one the score will be. This script will compute the 
PSNR for all pairs of images stored into ground truth folder and restored folder. The resuls will be saved in a .csv file. 

`python psnr.py --gt_folder *path/to/gt* --restored_folder *path/to/restored/images*` </br></br>


- 📝 **ssim.py** : Structural Similarity Index Metric [[1]](#1).

`python ssim.py --gt_folder *path/to/gt* --restored_folder *path/to/restored/images*` </br></br>


## Blind assessment

- 📝 **blockiness.py** : based on [[2]](#2) </br></br>


## Metrics evaluation

- 📝 **evaluation_functions.py** : </br></br>


- 📝 **evaluation_report.py** : </br></br>


## References

<a id="1">[1]</a> Wang, Zhou, et al. ["Image quality assessment: from error visibility to structural similarity."](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) </br>
IEEE transactions on image processing 13.4 (2004): 600-612.

<a id="2">[2]</a> Wang, Zhou, Alan C. Bovik, and Brian L. Evan. ["Blind measurement of blocking artifacts in images."](https://users.ece.utexas.edu/~bevans/courses/ee381k/projects/fall98/zhou/report.pdf)</br>
Proceedings 2000 International Conference on Image Processing (Cat. No. 00CH37101). Vol. 3. Ieee, 2000.
