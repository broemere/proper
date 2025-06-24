from collections import defaultdict
import csv
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import cv2
from file_handling import *


def frame_loader(signals, vid_file, frame_indices):
    vid = cv2.VideoCapture(vid_file)
    frame_count = int(vid.get(7))
    signals.message.emit("Collecting frame data")
    for i, f in enumerate(frame_indices):
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        res, frame = vid.read()
        if res:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for j in range(len(str(frame_count))):
                    gray[0, i] = int(str(frame_count)[j])
                gray[0, i + 1] = 10
                out_dir = str(make_dir(home / "images"))
                output_filename = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(vid_file))[0]}_{f}.png")
                cv2.imwrite(output_filename, gray)
                #yield i / len(frame_indices)
                pct = int((i/len(frame_indices)) * 100)
                signals.progress.emit(pct)
            except:
                vid.release()
    vid.release()
    #yield 1
    signals.progress.emit(100)


def load_csv(data_file):
    data = defaultdict(list)
    with open(data_file, mode='r', newline='') as f:
        reader = csv.reader(f)
        first_row = next(reader)
        p_col = 2
        other_col = []
        if "EMPTY" in first_row:
            next(reader)
            next(reader)
            p_col = 1
        else:
            if "putty log" not in first_row[0].lower():
                if len(first_row) > 1 and "pressure" in first_row[1].lower():
                    p_col = 1
                next(reader)

        for row in reader:
            if "pressure" in [cell.lower() for cell in row] or "sec" in [cell.lower() for cell in row]:
                continue
            if sum([cell != "" for cell in row]) > 1:
                if "(" in row[-1] or sum([cell != "" for cell in row]) == 2:
                    p_col = 1
                try:
                    data["t"].append(float(row[0]))
                except ValueError as ve:
                    continue
                try:
                    data["p"].append(float(row[p_col]))
                    if p_col == 2:
                        other_col.append(float(row[1]))
                except ValueError as ve:
                    data["t"].pop()
                    continue
            else:
                break

        if np.mean(np.gradient(data["p"])) == -1:
            data["p"] = other_col

    if len(data["t"]) == 0 or len(data["p"]) == 0:
        print("Missing data")
        print(data_file)
        print(f"t: {len(data['t'])}, p: {len(data['p'])}")
    if len(data["t"]) != len(data["p"]):
        print("Length mismatch")
        print(data_file)
        print(f"t: {len(data['t'])}, p: {len(data['p'])}")

    data = dict(data)
    x_data = np.arange(len(data["t"])).reshape(-1, 1)
    ransac = RANSACRegressor(estimator=LinearRegression(), random_state=0)
    ransac.fit(x_data, data["t"])
    y_pred = ransac.predict(x_data)
    y_pred = np.round(y_pred, 1) + 0

    return {"t": y_pred.tolist(), "p": data["p"],}
