from processing.data_transform import zero_data, smooth_data
from processing.data_loader import load_csv, frame_loader
import numpy as np
import logging

log = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, settings={}):
        self.raw_data = {"t": [], "p": []}
        self.trimmed_data = {"t": [], "p": []}
        self.zeroed_data = {"t": [], "p": []}
        self.smoothed_data = {"t": [], "p": []}
        self.csv_path = None
        self.length = 42
        self.video = None
        self.initial_pressure = 0
        self.initial_index = 0
        self.final_pressure = 25
        self.final_index = -1
        self.brightness = 50
        self.contrast = 50
        self.threshold = 127
        self.smoothing_method = "Min"
        self.smoothing_window = 100
        self.zeroing_method = "Min"
        self.zeroing_window = 7
        self.trim_start = 0
        self.trim_stop = 42
        self.known_length = 0
        self.conversion_factor = 0
        self._observers = {}
        self.task_manager = None
        self.__dict__.update(settings)

        self.baseline_image_first: np.ndarray = None
        self.transformed_image_first: np.ndarray = None
        self.threshed_image_first: np.ndarray = None

        self.baseline_image: np.ndarray = None
        self.transformed_image: np.ndarray = None
        self.threshed_image: np.ndarray = None

        self.scale_frame = 21
        self.scale_image: np.ndarray = None


    def register_observer(self, key: str, callback):
        """Register a callback to be invoked when the given key changes."""
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(callback)

    def notify_observers(self, key: str, value):
        """Notify all observers that a particular key has changed."""
        for callback in self._observers.get(key, []):
            callback(value)

    def update_pipeline(self):
        """
        Run the entire pipeline: apply trimming, then smoothing, then zeroing.
        This ensures that any change in a parameter cascades through the pipeline.
        """
        # 1. Apply trimming.
        self.trimmed_data = {key: self.raw_data[key][max(0, int(self.trim_start)):min(int(self.trim_stop), self.length)] for key in self.raw_data}
        # 2. Apply zeroing on the trimmed data (using the current smoothing parameters too).
        self.zeroed_data = zero_data(self.trimmed_data, self.zeroing_method, self.zeroing_window + 1)
        # 3. Apply smoothing on the trimmed data.
        self.smoothed_data = smooth_data(self.zeroed_data, self.smoothing_method, self.smoothing_window + 1)
        self.notify_observers('transformed', [self.zeroed_data, self.smoothed_data])

    def load_csv_file(self, file_path: str):
        """
        Load a CSV file, initialize the data stages, and compute a hash for the file.
        """
        self.csv_path = file_path
        self.raw_data = load_csv(file_path)
        self.length = int(len(self.raw_data["t"]))
        self.trim_stop = self.length
        self.notify_observers('raw', None)
        self.update_pipeline()

    def set_trimming(self, start: int, stop: int):
        """
        Set the trimming parameters and schedule an update.
        """
        self.trim_start = start
        self.trim_stop = stop
        self.update_pipeline()

    def set_zeroing(self, method: str, window: int):
        self.zeroing_method = method
        self.zeroing_window = window
        self.update_pipeline()

    def set_smoothing(self, method: str, window: int):
        self.smoothing_method = method
        self.smoothing_window = window
        self.update_pipeline()

    def get_data(self, data_version: str):
        return getattr(self, f"{data_version}_data", {})

    # def load_video_file(self, signals):
    #     frame_loader(signals, self.video, [self.trim_start, self.final_index])
    def load_video_file(self, file_path: str):
        self.video = file_path
        self.scale_frame = int((self.trim_stop + self.trim_start) / 2)

        #frame_loader(None, self.video, [middle_frame])
        self.task_manager.queue_task(
            frame_loader,  # The function to run
            self.video,  # This will be the 'vid_file' argument
            [self.scale_frame],  # This will be the 'frame_indices' argument
            on_result=self.scale_frame_loaded # Optional: a method in DataPipeline to handle the result
        )
        #self.notify_observers('video', None)

    def scale_frame_loaded(self, result: dict):
        """
        Callback function executed when the frame_loader task is complete.
        'result' is the dictionary of NumPy arrays returned by frame_loader.
        """
        log.info(f"Received {len(result)} loaded frames from worker.")

        # Let's say we requested one frame for the scaling tab.
        # We need to know which frame we got. The dictionary keys help here.
        if not result:
            log.warning("Frame loader returned no frames.")
            return

        # Example: Get the first (and likely only) frame from the result.
        first_frame_index = next(iter(result))
        self.scale_image = result[first_frame_index]
        # THE MOST IMPORTANT STEP: Notify the UI that new data is ready!
        self.notify_observers('scale_image', self.scale_image)

    def set_images(self, image_array_first: np.ndarray, image_array: np.ndarray):
        """Store the original frame and initialize transformed_image to match."""
        self.baseline_image_first = image_array_first.copy()
        self.transformed_image_first = image_array_first.copy()
        self.threshed_image_first = image_array_first.copy()
        self.baseline_image = image_array.copy()
        self.transformed_image = image_array.copy()
        self.threshed_image = image_array.copy()
        self.notify_observers('bcimage', [self.transformed_image_first, self.transformed_image])
        self.notify_observers('thimage', [self.threshed_image_first, self.threshed_image])

    def apply_leveling(self):
        """Recompute transformed_image from baseline_image using current self.brightness and self.contrast (0–100)"""
        if self.baseline_image is None:
            return

        img_first = self.baseline_image_first.astype(np.float32)
        img = self.baseline_image.astype(np.float32)
        default_min, default_max = 0.0, 255.0
        full_range = default_max - default_min  # 255
        slider_range = 100.0
        mid = slider_range / 2.0  # 50
        # invert so slider=100 → newCenter=0, slider=0 → newCenter=255
        inv_b = slider_range - self.brightness
        new_center = default_min + full_range * (inv_b / slider_range)
        # contrast slope like ImageJ
        eps = 1e-4
        c = float(self.contrast)
        if c <= mid:
            slope = c / mid
        else:
            slope = mid / ((slider_range - c) + eps)
        new_min = new_center - (0.5 * full_range) / slope
        new_max = new_center + (0.5 * full_range) / slope
        denom = (new_max - new_min) if (new_max - new_min) != 0 else eps

        # linear map [new_min,new_max] → [0,255], per‐channel
        leveled_first = ((img_first - new_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        self.transformed_image_first = leveled_first
        leveled = ((img - new_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        self.transformed_image = leveled
        self.notify_observers('bcimage', [self.transformed_image_first, self.transformed_image])

    def apply_thresh(self):
        """
        Apply a binary threshold to the *transformed_image*, not baseline.
        Pixels >= self.threshold become 255; others 0.
        """
        if self.transformed_image is None:
            return

        img_first = self.transformed_image_first.astype(np.uint8)
        th = self.threshold  # (for a 2D grayscale image)
        th_img_first = np.where(img_first >= th, 255, 0).astype(np.uint8)
        self.threshed_image_first = th_img_first

        img = self.transformed_image.astype(np.uint8)
        th_img = np.where(img >= th, 255, 0).astype(np.uint8)
        self.threshed_image = th_img

        self.notify_observers('thimage', [self.threshed_image_first, self.threshed_image])

