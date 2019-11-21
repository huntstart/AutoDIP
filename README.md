# AutoDIP
This is the code part of [Towards the Automation of Deep Image Prior](https://arxiv.org/abs/1911.07185).
The code is based on [Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior).

## Install
Here is the list of libraries you need to install to execute the code:
- python = 3.6
- pytorch = 1.0
- numpy
- scipy
- matplotlib
- scikit-image

## One example
1. `python denoising.py`
2. The results are in `../denoise-set1`, Source images are in `../data/denoise.list` as default.
3. `python rotate_cur.py` 
4. In default, it will go to floder `../denoise-set1` to find folders of images listed in '../data/denoise.list'. The results are in `../denoise-set1/best-eopch.txt`
