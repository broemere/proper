from statistics import mean, median
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, median_filter
import cv2
from PySide6.QtGui import QImage, QPixmap
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb
import logging
log = logging.getLogger(__name__)


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

    # elif method == "Moving Avg":
    #     half_window = r // 2
    #     new_p = []
    #     for i in range(n):
    #         start = max(0, i - half_window)
    #         end = min(n, i + half_window + 1)  # +1 because the slice end is exclusive
    #         new_p.append(mean(data["p"][start:end]))
    # elif method == "Median":
    #     new_p = []
    #     half_window = r // 2
    #     for i in range(n):
    #         start = max(0, i - half_window)
    #         end = min(n, i + half_window + 1)
    #         new_p.append(median(data["p"][start:end]))

    # elif method == "Moving Avg":
    #     # mode='nearest' repeats the edge value
    #     # mode='mirror' reflects the data pattern at the edge
    #     new_p = uniform_filter1d(data["p"], size=r, mode='nearest')
    #
    # elif method == "Median":
    #     new_p = median_filter(data["p"], size=r, mode='nearest')

    elif method == "Moving Avg":
        half_window = r // 2
        new_p = []
        for i in range(n):
            # Calculate how much space we have on left vs right
            # and pick the smaller one to ensure symmetry.
            w = min(half_window, i, n - 1 - i)
            start = i - w
            end = i + w + 1  # +1 for exclusive slice
            new_p.append(mean(data["p"][start:end]))

    elif method == "Median":
        half_window = r // 2
        new_p = []
        for i in range(n):
            # symmetric logic
            w = min(half_window, i, n - 1 - i)
            start = i - w
            end = i + w + 1
            new_p.append(median(data["p"][start:end]))
    elif method == "Gaussian":
        new_p = np.asarray(data["p"], dtype=float)
        sigma = r / 6.0  # A common heuristic.
        new_p = gaussian_filter1d(new_p, sigma=sigma, mode='nearest').tolist()
    else:
        #return ui.notify("SMOOTHING ERROR: INCORRECT METHOD")
        print("Smoothing Error")
    return {"t": data["t"], "p": new_p}


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
    return {"t": data["t"], "p": np.clip([round(x, 2) for x in new_p], 0, None)}


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


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """
    Converts a QImage to a NumPy array with detailed logging and format handling.
    """
    if qimg.isNull():
        log.warning("Received a null QImage.")
        return np.array([], dtype=np.uint8)

    # Step 1: Pick a consistent format
    log.debug("Step 1: Picking image format...")
    fmt = qimg.format()
    if fmt == QImage.Format_Grayscale8:
        target_fmt, n_chan = QImage.Format_Grayscale8, 1
    else:
        has_alpha = qimg.hasAlphaChannel()
        target_fmt, n_chan = (QImage.Format_RGBA8888, 4) if has_alpha else (QImage.Format_RGB888, 3)

    if fmt != target_fmt:
        qimg = qimg.convertToFormat(target_fmt)
    log.debug(f"Step 1 Complete. Target format n_chan: {n_chan}")

    # Step 2: Accessing raw data buffer
    log.debug("Step 2: Accessing raw data buffer (constBits)...")
    buf = qimg.constBits()
    log.debug("Step 2 Complete.")

    # Step 3: Creating NumPy array from buffer
    log.debug("Step 3: Creating NumPy array from buffer...")
    arr = np.frombuffer(buf, dtype=np.uint8)
    log.debug("Step 3 Complete.")

    # Step 4: Reshaping array
    log.debug("Step 4: Reshaping array...")
    height, width, bpl = qimg.height(), qimg.width(), qimg.bytesPerLine()
    arr = arr.reshape((height, bpl))
    line_stride = width * n_chan
    arr = arr[:, :line_stride]
    if n_chan > 1:
        arr = arr.reshape((height, width, n_chan))
    else:
        arr = arr.reshape((height, width))
    log.debug(f"Step 4 Complete. Array shape: {arr.shape}")

    # Step 5: Grayscale Check
    if n_chan in [3, 4]:
        log.debug("Step 5: Checking for monochrome content...")
        # Check if the first channel is equal to the second, and the second to the third.
        is_grayscale = np.array_equal(arr[:, :, 0], arr[:, :, 1]) and \
                       np.array_equal(arr[:, :, 1], arr[:, :, 2])
        if is_grayscale:
            log.debug("Monochrome content detected. Returning single channel.")
            return arr[:, :, 0].copy()  # Return a copy to be safe
        log.debug("Step 5 Complete. Content is color.")

    # Step 6: This is optional, but keeps consistency if your pipeline expects BGR
    if n_chan > 1:
        log.debug("Step 6: Converting color format with OpenCV...")
        if n_chan == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif n_chan == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        log.debug("Step 6 Complete.")

    return arr.copy()  # Return a copy to ensure data is independent

def get_kernel_size(l):
    k = round(l ** 0.2)
    if k % 2 == 0:
        k = k + 1
    return int(k)

def label_image(signals, arr, left_right):
    try:
        log.info("Worker starting: label_image task.")

        kernel_size = get_kernel_size(arr.shape[0])

        log.info(f"Kernel size: {kernel_size}")

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Call morphologyEx without the extra argument
        arr_opened = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)

        # The conversion is already done, so we just label the array
        labels = label(arr_opened, background=0, connectivity=2)

        log.info("Worker finished: label_image task successful.")

        return [labels, left_right]

    except Exception:
        log.exception("An unhandled exception occurred in the label_image worker!")
        return None


def create_visual_from_labels(signals, labels, left_right, min_size=None):
    """
    Filters small blobs from a labels array and converts it to a color image,
    reporting progress along the way.
    """
    try:
        signals.message.emit("Analyzing labeled regions...")

        if min_size == None:
            min_size = np.sqrt(labels.shape[0]*labels.shape[1])
            log.info(f"Minimum viable area: {min_size} pix")

        filtered_labels = labels.copy()

        # Find the unique labels and their sizes (excluding background)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_sizes = {label: count for label, count in zip(unique_labels, counts) if label != 0}

        # Get a list of labels to check for progress reporting
        labels_to_check = list(label_sizes.keys())
        total_to_check = len(labels_to_check)

        n_labels = 0

        if total_to_check > 0:
            signals.message.emit(f"Analyzing {total_to_check} blobs...")
            for i, label_id in enumerate(labels_to_check):
                # Filter blobs smaller than the threshold
                if label_sizes[label_id] < min_size:
                    filtered_labels[filtered_labels == label_id] = 0
                else:
                    n_labels += 1

                # Emit progress every 5% to avoid flooding the UI
                pct = int(((i + 1) / total_to_check) * 100)
                if i == 0 or (pct % 5 == 0 and int((i / total_to_check) * 100) != pct):
                    signals.progress.emit(pct)

        signals.message.emit("Calculating blob properties...")
        if n_labels > 0:
            # Only call regionprops_table if there are labels to analyze
            properties_to_measure = [
                'label', 'area', 'centroid', 'perimeter', 'solidity', 'extent',
                'major_axis_length', 'minor_axis_length', 'eccentricity',
                'orientation', 'moments_hu'
            ]
            props = regionprops_table(filtered_labels, properties=properties_to_measure)
        else:
            # If no labels, create an empty dictionary to maintain data structure
            log.info("No viable labels found after filtering. Skipping regionprops.")
            props = {} # or an empty pandas DataFrame if you expect one

        signals.message.emit("Generating color image...")
        color_image = label2rgb(filtered_labels, bg_label=0, bg_color=(0, 0, 0))
        color_image_uint8 = (color_image * 255).astype(np.uint8)

        signals.progress.emit(100)
        signals.message.emit("Visualization complete.")
        log.info(f"Found {n_labels} viable labels and {total_to_check} total labels")

        # Return both the original labels (for analysis) and the new visual
        return {'labels': filtered_labels, 'visual': color_image_uint8, 'left_right': left_right, 'properties': props}

    except Exception as e:
        log.exception("Error in create_visual_from_labels worker!")
        signals.message.emit("Error during visualization.")
        return {'labels': labels, 'visual': None}  # Return a dict to maintain structure

def convert_numpy(obj):
    """
    Recursively convert numpy arrays and scalars into JSON-serializable forms:
      - ndarray → dict with keys __ndarray__, dtype, shape, data (as nested lists)
      - numpy scalar → native Python type via .item()
    """
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": obj.shape
        }
    elif isinstance(obj, np.generic):
        # covers np.int32, np.float64, etc.
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(v) for v in obj)
    else:
        return obj


def restore_numpy(obj):
    """
    Recursively walk obj and convert back to numpy arrays when possible:
      - If obj is a dict with "__ndarray__", build an array with the right dtype & shape.
      - If obj is a nested list of numbers (1D/2D/3D), cast it to an ndarray.
      - Otherwise, recurse into dicts & lists; leave other types untouched.
    """
    # 1) Reverse of our special-encoded ndarray
    if isinstance(obj, dict) and "__ndarray__" in obj:
        arr = np.array(obj["__ndarray__"], dtype=np.dtype(obj.get("dtype", None)))
        if "shape" in obj:
            arr = arr.reshape(obj["shape"])
        return arr

    # 2) Try to turn pure numeric lists into arrays
    if isinstance(obj, list):
        try:
            arr = np.array(obj)
            # only accept if it really is numeric and 1–3 dimensional
            if arr.dtype.kind in ("i","u","f") and 1 <= arr.ndim <= 3:
                return arr
        except Exception:
            pass
        # otherwise, recurse into each element
        return [restore_numpy(v) for v in obj]

    # 3) Recurse into plain dicts
    if isinstance(obj, dict):
        return {k: restore_numpy(v) for k, v in obj.items()}

    # 4) Leave everything else alone
    return obj

def format_value(value: float) -> str:
    """
    Formats a number with a variable number of decimal places based on its magnitude.
    - 3 decimal places for numbers < 1
    - 2 decimal places for numbers < 10
    - 1 decimal place for numbers >= 10
    """
    # Gracefully handle cases where the value might not be a number
    if not isinstance(value, (int, float)):
        return str(value)
    if value == int(value):
        return f"{int(value)}" # Format as a simple integer

    num_abs = abs(value)
    if num_abs < 1:
        precision = 3
    elif num_abs < 10:
        precision = 2
    else:  # For numbers 10 and greater
        precision = 1
    return f"{value:.{precision}f}"

def n_closest_numbers(nums, n):
    if n > len(nums):
        return nums

    # Step 1: sort the list
    nums = sorted(nums)

    min_range = float('inf')
    best_group = []

    # Step 2: slide a window of size n
    for i in range(len(nums) - n + 1):
        window = nums[i:i + n]
        spread = window[-1] - window[0]  # max - min in the window
        if spread < min_range:
            min_range = spread
            best_group = window

    return best_group
