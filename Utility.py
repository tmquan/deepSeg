#from __future__ import print_function

import mxnet as mx
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

import skimage.io
import pydot
import cv2
import sys
import logging
import os
import dicom
import scipy
import re 
import natsort

import csv
import random
import subprocess
from matplotlib.pyplot import ion
from sklearn.cross_validation import KFold # For cross_validation
from sklearn.metrics import mean_squared_error
from skimage.restoration import denoise_tv_chambolle
from scipy.misc import imresize
from scipy import ndimage
from scipy.stats import norm
from keras.utils.generic_utils import Progbar
from multiprocessing import Process
from joblib import Parallel, delayed
from random import randint
from graphviz import Digraph 