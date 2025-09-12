from collections import defaultdict
import csv
import numpy as np
import cv2
import logging
import os
import getpass
from tifffile import TiffFile

log = logging.getLogger(__name__)


def frame_loader(signals, vid_file, frame_indices, count=False):
    """
    Loads specific frames from a video or multi-page TIFF file, handling errors gracefully.

    This function checks the file extension to determine the loading method. For TIFF files,
    it uses the tifffile library to directly access frames by index. For all other file
    types, it uses OpenCV's VideoCapture.
    """
    loaded_frames = {}
    file_path = str(vid_file)
    # Determine the file type by its extension
    file_ext = os.path.splitext(file_path)[1].lower()

    # --- TIFF File Handling ---
    if file_ext in ['.tif', '.tiff']:
        try:
            # Use a 'with' statement for safe file handling
            with TiffFile(file_path) as tif:
                frame_count = len(tif.pages)
                signals.message.emit("Collecting TIFF image data...")
                log.info(f"Starting frame extraction for {len(frame_indices)} frames from {file_path}")

                for i, f in enumerate(frame_indices):
                    try:
                        # Check if the requested frame index is valid
                        if f >= frame_count:
                            log.warning(
                                f"Frame index {f} is out of bounds for TIFF with {frame_count} pages. Skipping.")
                            continue

                        # Directly access the specific page (frame) by its index
                        frame = tif.pages[f].asarray()

                        # Ensure the frame is grayscale for consistent processing
                        if frame.ndim == 3:
                            # Handles RGB or multi-channel images
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            # It's already grayscale, create a copy
                            gray = frame.copy()

                        # Normalize to 8-bit uint for compatibility with downstream processes
                        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
                        gray = gray.astype(np.uint8)

                        # Embed frame count into the first row of pixels
                        if i + len(str(frame_count)) < gray.shape[1]:
                            for j, digit in enumerate(str(frame_count)):
                                gray[0, i + j] = int(digit)
                        else:
                            log.warning(f"Frame {f}: Not enough horizontal pixels to write metadata.")

                        loaded_frames[f] = gray

                        pct = int(((i + 1) / len(frame_indices)) * 100)
                        signals.progress.emit(pct)

                    except Exception as e:
                        # Log error for a single frame and continue with the next
                        log.error(f"Error processing TIFF frame at index {f}: {e}", exc_info=True)
                        signals.message.emit(f"Error on TIFF frame {f}, see log for details.")

                if count:
                    loaded_frames[frame_count] = None

        except FileNotFoundError:
            err_msg = f"TIFF file not found: {file_path}"
            log.error(err_msg)
            signals.message.emit(err_msg)
            raise IOError(err_msg)
        except Exception as e:
            # Catch other potential errors like a corrupted TIFF file
            err_msg = f"Failed to open or process TIFF file: {file_path}. Error: {e}"
            log.error(err_msg, exc_info=True)
            signals.message.emit(err_msg)
            raise IOError(err_msg)

    # --- Video File Handling (Original Logic) ---
    else:
        vid = cv2.VideoCapture(file_path)
        if not vid.isOpened():
            err_msg = f"Failed to open video file: {file_path}"
            log.error(err_msg)
            signals.message.emit(err_msg)
            raise IOError(err_msg)

        try:
            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            signals.message.emit("Collecting frame data...")
            log.info(f"Starting frame extraction for {len(frame_indices)} frames from {file_path}")

            for i, f in enumerate(frame_indices):
                try:
                    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
                    res, frame = vid.read()

                    if not res:
                        log.warning(f"Could not read frame at index {f}. Skipping.")
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)

                    # Embed frame count using the same logic as for TIFFs
                    if i + len(str(frame_count)) < gray.shape[1]:
                        for j, digit in enumerate(str(frame_count)):
                            gray[0, i + j] = int(digit)
                    else:
                        log.warning(f"Frame {f}: Not enough horizontal pixels to write metadata.")

                    loaded_frames[f] = gray

                    pct = int(((i + 1) / len(frame_indices)) * 100)
                    signals.progress.emit(pct)

                except Exception as e:
                    log.error(f"Error processing frame at index {f}: {e}", exc_info=True)
                    signals.message.emit(f"Error on frame {f}, see log for details.")

            if count:
                loaded_frames[frame_count] = None
        finally:
            # Ensure the video capture is always released
            log.info(f"Finished frame extraction. Releasing video capture for {file_path}.")
            vid.release()

    # --- Finalization (Common to Both Paths) ---
    signals.progress.emit(100)
    signals.message.emit("Frame processing complete.")
    return loaded_frames


def load_csv(data_file):
    data = defaultdict(list)
    pressures = []
    times = []
    with open(data_file, mode='r', newline='') as f:
        reader = csv.reader(f)
        row = next(reader)
        tcol = None
        pcol = None

        if len(row) == 1:
            row = next(reader)
        else:
            for col in row:
                if "putty" in col.lower():
                    row = next(reader)
                    break
                if "empty" in col.lower():
                    row = next(reader)
                    row = next(reader)
                    break

        lowered = [col.lower().strip() for col in row]
        pcol = next((i for i, col in enumerate(lowered) if col == "pressure"), None)
        t_candidates = ["msec", "sec", "devicetime", "time"]
        for key in t_candidates:
            try:
                tcol = lowered.index(key)
                break
            except ValueError:
                continue

        if pcol is not None and tcol is not None:
            for row in reader:
                try:
                    pressures.append(float(row[pcol]))
                except (IndexError, ValueError):
                    pressures.append(pressures[-1])
                try:
                    times.append(float(row[tcol]))
                except (IndexError, ValueError):
                    times.append(times[-1])
        else:
            for row in reader:
                for i, col in enumerate(row):
                    if "(" not in col:
                        data[i].append(float(col))
            bitcol = None
            for k in data.keys():
                median_grad = np.median(np.gradient(data[k]))
                if 1.1 > median_grad > 0.9:
                    bitcol = k
            if bitcol is not None:
                del data[bitcol]
            rs = defaultdict(list)
            for k in data.keys():
                r = np.corrcoef(np.arange(0, len(data[k])), data[k])
                rs[k].append(r[0][1])
            tcol = max(rs, key=rs.get)
            pcol = min(rs, key=rs.get)

            pressures = data[pcol]
            times = data[tcol]

        times_fixed = np.arange(0, len(times)) * np.median(np.gradient(times)).tolist()
        times_rounded = [round(x, 2) for x in times_fixed]

        data = {"t": times_rounded, "p": pressures}
    return data

def get_system_username():
    """
    Returns the current system user name in a cross-platform way.
    """
    try:
        # Primary method
        return getpass.getuser()
    except Exception:
        # Fallbacks: Windows, Unix, etc.
        return (
            os.environ.get('USERNAME')
            or os.environ.get('USER')
            or os.environ.get('LOGNAME')
            or None
        )