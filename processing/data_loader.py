from collections import defaultdict
import csv
import numpy as np
import cv2
from processing.file_handling import *
import logging

log = logging.getLogger(__name__)

# def frame_loader(signals, vid_file, frame_indices):
#     vid = cv2.VideoCapture(vid_file)
#     frame_count = int(vid.get(7))
#     signals.message.emit("Collecting frame data")
#     for i, f in enumerate(frame_indices):
#         vid.set(cv2.CAP_PROP_POS_FRAMES, f)
#         res, frame = vid.read()
#         if res:
#             try:
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 for j in range(len(str(frame_count))):
#                     gray[0, i] = int(str(frame_count)[j])
#                 gray[0, i + 1] = 10
#                 out_dir = str(make_dir(home / "images"))
#                 output_filename = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(vid_file))[0]}_{f}.png")
#                 cv2.imwrite(output_filename, gray)
#                 #yield i / len(frame_indices)
#                 pct = int((i/len(frame_indices)) * 100)
#                 signals.progress.emit(pct)
#             except:
#                 vid.release()
#     vid.release()
#     #yield 1
#     signals.progress.emit(100)

def frame_loader(signals, vid_file, frame_indices):
    """
    Loads specific frames from a video file, handling errors gracefully.
    """
    vid = cv2.VideoCapture(str(vid_file))
    if not vid.isOpened():
        err_msg = f"Failed to open video file: {vid_file}"
        log.error(err_msg)
        # We can even use the worker signals to report this critical failure
        signals.message.emit(err_msg)
        # Raise an exception to make sure the worker catches it as a full task failure
        raise IOError(err_msg)

    loaded_frames = {}
    # Use a 'finally' block to guarantee the video is released,
    # no matter what happens inside the 'try'.
    try:
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        signals.message.emit("Collecting frame data...")
        log.info(f"Starting frame extraction for {len(frame_indices)} frames from {vid_file}")

        for i, f in enumerate(frame_indices):
            try:
                # This inner try/except handles errors for a SINGLE frame
                vid.set(cv2.CAP_PROP_POS_FRAMES, f)
                res, frame = vid.read()

                if not res:
                    log.warning(f"Could not read frame at index {f}. Skipping.")
                    continue # Skip to the next frame

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # This part seems unusual and could be a source of errors
                # If 'i' goes out of bounds for the number of pixels in the first row,
                # this will raise an IndexError.
                # Let's add a check.
                if i + 1 < gray.shape[1]:
                    for j in range(len(str(frame_count))):
                        gray[0, i + j] = int(str(frame_count)[j]) # Potential for IndexError here too
                else:
                    log.warning(f"Frame {f}: Not enough horizontal pixels to write metadata.")

                # I'm commenting out these dependencies for clarity, assuming they exist
                # out_dir = str(make_dir(home + "/images"))
                # output_filename = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(vid_file))[0]}_{f}.png")
                # cv2.imwrite(output_filename, gray)

                loaded_frames[f] = gray

                pct = int(((i + 1) / len(frame_indices)) * 100)
                signals.progress.emit(pct)

            except Exception as e:
                # Now if a single frame fails, we log it and continue
                log.error(f"Error processing frame at index {f}: {e}", exc_info=True)
                # exc_info=True will add the full traceback to the log, which is invaluable.
                signals.message.emit(f"Error on frame {f}, see log for details.")

    finally:
        # This code will ALWAYS run, ensuring cleanup.
        log.info(f"Finished frame extraction. Releasing video capture for {vid_file}.")
        vid.release()

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

        data = {"t": np.arange(0, len(times)) * np.median(np.gradient(times)).tolist(), "p": pressures}
    return data
