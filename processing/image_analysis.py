import cv2
import numpy as np
from skimage.measure import label, regionprops


def get_blobs(img, connectivity=2):
    labels = label(img, background=0, connectivity=connectivity)
    return labels

def blob_properties(labels):
    props = regionprops(labels)
    return props

def morph(img, mode, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
    out = cv2.morphologyEx(img, mode, kernel)
    return out

def opening(img, kernel_size):
    return morph(img, cv2.MORPH_OPEN, kernel_size)

def closing(img, kernel_size):
    return morph(img, cv2.MORPH_CLOSE, kernel_size)

def dilate(img, kernel_size):
    return morph(img, cv2.MORPH_DILATE, kernel_size)

def erode(img, kernel_size):
    return morph(img, cv2.MORPH_ERODE, kernel_size)

def threshold(img, th):
    _, thresh = cv2.threshold(img, 0, th, cv2.THRESH_BINARY)
    return thresh

