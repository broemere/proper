# @author: broemere
# created: 1/12/2022

from __future__ import with_statement
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statistics import median, mean, mode
from copy import deepcopy
import matplotlib.transforms as mtrans
from types import SimpleNamespace as ns
from pathlib import Path
import plotly.graph_objects as go
#from statsmodels.nonparametric.kernel_regression import KernelReg
import statsmodels.api as sm
from numpy.linalg import eig, inv
import scipy.stats
from scipy.spatial import ConvexHull
from alive_progress import alive_bar
from numba import jit, cuda

import errno
import winsound
from time import perf_counter, sleep
import pickle
import datetime
from shutil import copytree, rmtree

from filecmp import dircmp
import itertools, functools, operator
from zipfile import ZipFile
import xmltodict
from openpyxl import load_workbook
import multiprocessing as mp
import queue
from skimage.filters import threshold_otsu

from skimage.color import rgb2gray
from skimage.filters import sobel, roberts
from skimage.feature import canny
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

import trimesh
from skimage import measure
from skimage.draw import ellipsoid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dateutil.parser import parse as parsedate

import humanize

import pprint


def main():
    pass

if __name__ == "__main__":
    main()