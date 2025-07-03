from statistics import mean, median
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
from PySide6.QtGui import QImage, QPixmap


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


def zero_data(data, method, r):
    if method == "None":
        return data
    if method == "First":
        zero = data["p"][0]
    elif method in ("Min", "Mean", "Median"):
        subset = data["p"][0:r]
        if method == "Min":
            zero = min(subset)
        elif method == "Mean":
            zero = mean(subset)
        elif method == "Median":
            zero = median(subset)
    else:
        #return ui.notify("ZEROING ERROR: INCORRECT METHOD")
        print("Zeroing Error")
    new_p = [round(p - zero, 2) for p in data["p"]]
    return {"t": data["t"], "p": new_p}


def numpy_to_qpixmap(numpy_array: np.ndarray) -> QPixmap:
    """
    Converts a NumPy array to a QPixmap.

    Handles both grayscale (2D) and color (3D) images.
    Assumes color images from OpenCV are in BGR format.
    """
    if numpy_array is None:
        return QPixmap()  # Return an empty pixmap if the array is null

    height, width = numpy_array.shape[:2]
    bytes_per_line = numpy_array.strides[0]

    # --- Determine the QImage format ---
    if numpy_array.ndim == 2:
        # Grayscale image
        q_image_format = QImage.Format_Grayscale8
    elif numpy_array.ndim == 3:
        # Color image
        if numpy_array.shape[2] == 4:
            # RGBA format
            q_image_format = QImage.Format_RGBA8888
        else:
            # Standard 3-channel color. OpenCV uses BGR, but Qt needs RGB.
            # We must convert it.
            numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
            q_image_format = QImage.Format_RGB888
    else:
        # Unsupported format
        return QPixmap()

    # --- Create QImage from the NumPy array's memory buffer ---
    q_image = QImage(numpy_array.data, width, height, bytes_per_line, q_image_format)

    # QImage might hold a reference to the numpy array. To be safe,
    # copy it before returning, so the array can be garbage collected.
    return QPixmap.fromImage(q_image.copy())

