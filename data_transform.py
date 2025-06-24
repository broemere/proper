from statistics import mean, median
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2


def brightness_contrast(img, brightness, contrast):
    """
    Adjust brightness and contrast of an image using OpenCV.

    Parameters:
      img (np.ndarray): The input image in BGR format.
      brightness (float): Brightness value between 0 and 100.
      contrast (float): Contrast value between 0 and 100.

    Returns:
      np.ndarray: The adjusted image.
    """
    # Define default parameters.
    defaultMin = 0
    defaultMax = 255
    range_val = defaultMax - defaultMin  # 255
    sliderRange = 100.0
    mid = sliderRange / 2.0  # 50

    # Invert brightness so that increasing the slider makes the image brighter.
    invBrightness = sliderRange - brightness

    # Compute new display center (window level) from inverted brightness.
    newCenter = defaultMin + (defaultMax - defaultMin) * (invBrightness / sliderRange)

    # Compute contrast slope as per ImageJ.
    epsilon = 0.0001  # To avoid division by zero.
    if contrast <= mid:
        slope = contrast / mid
    else:
        slope = mid / ((sliderRange - contrast) + epsilon)

    # Compute new display window based on the computed slope.
    new_min = newCenter - (0.5 * range_val) / slope
    new_max = newCenter + (0.5 * range_val) / slope

    # Convert image to float for precise computation.
    img_float = img.astype(np.float32)

    # Apply the linear mapping for each pixel.
    adjusted = (img_float - new_min) / (new_max - new_min) * 255.0
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    return adjusted


def find_binary_th(img):
    for i in np.arange(1, 254):
        _, th = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)
        hi = img[np.where(img > i)]
        lo = img[np.where(img <= i)]

        comp_avg = (np.mean(hi[np.where(hi < 255)]) + np.mean(lo[np.where(lo > 0)])) / 2

        if not np.isnan(comp_avg) and i + 1 >= round(comp_avg):
            #print(i, comp_avg)
            break

    return i


def smooth_data(data, method, r):
    n = len(data["t"])
    #cache = json_load(".nicegui/storage-general.json")
    #method = cache["smoothing"]
    #r = cache["smooth_range"] + 1
    if method == "None":
        return data
    elif method == "Min":
        new_p = [min(data["p"][i:min(i + r, n)]) for i in range(n)]
    elif method == "Double Min":
        first_smooth = [min(data["p"][i:min(i + r, n)]) for i in range(n)]
        new_p = [min(first_smooth[i:min(i + r, n)]) for i in range(n)]
    elif method == "Moving Avg":
        half_window = r // 2
        new_p = []
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)  # +1 because the slice end is exclusive
            new_p.append(mean(data["p"][start:end]))
    elif method == "Median":
        new_p = []
        half_window = r // 2
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            new_p.append(median(data["p"][start:end]))
    elif method == "Gaussian":
        new_p = np.asarray(data["p"], dtype=float)
        sigma = r / 6.0  # A common heuristic.
        new_p = gaussian_filter1d(new_p, sigma=sigma, mode='nearest').tolist()
    else:
        #return ui.notify("SMOOTHING ERROR: INCORRECT METHOD")
        print("Smoothing Error")
    return {"t": data["t"], "p": [round(x, 2) for x in new_p]}


def zero_data(data, method, r, smooth_method, smooth_r):
    new_data = smooth_data(data, smooth_method, smooth_r)
    #cache = json_load(".nicegui/storage-general.json")
    #method = cache["zeroing"]
    #r = cache["zero_range"] + 1
    if method == "None":
        return new_data

    if method == "First":
        zero = new_data["p"][0]
    elif method in ("Min", "Mean", "Median"):
        subset = new_data["p"][0:r]
        if method == "Min":
            zero = min(subset)
        elif method == "Mean":
            zero = mean(subset)
        elif method == "Median":
            zero = median(subset)
    else:
        #return ui.notify("ZEROING ERROR: INCORRECT METHOD")
        print("Zeroing Error")

    new_p = [round(p - zero, 2) for p in new_data["p"]]
    return {"t": new_data["t"], "p": new_p}
