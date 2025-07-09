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
        self.frame_count = 0
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
        self.left_image: np.ndarray = None
        self.right_image: np.ndarray = None


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
        self.trimmed_data = {key: self.raw_data[key][max(0, int(self.trim_start)):min(int(self.trim_stop+1), self.length)] for key in self.raw_data}
        # 2. Apply zeroing on the trimmed data (using the current smoothing parameters too).
        self.zeroed_data = zero_data(self.trimmed_data, self.zeroing_method, self.zeroing_window + 1)
        # 3. Apply smoothing on the trimmed data.
        self.smoothed_data = smooth_data(self.zeroed_data, self.smoothing_method, self.smoothing_window + 1)
        self.notify_observers('transformed', [self.zeroed_data, self.smoothed_data])
        print("Points plotted:", len(self.smoothed_data["p"]))

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
        self.notify_observers('trimming', (start, stop))

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
            True,
            on_result=self.scale_frame_loaded # Optional: a method in DataPipeline to handle the result
        )
        #self.notify_observers('video', None)

    def scale_frame_loaded(self, result: dict):
        """
        Callback function executed when the frame_loader task is complete.
        'result' is the dictionary of NumPy arrays returned by frame_loader.
        """
        log.info(f"Received {len(result)-1} loaded frames from worker.")

        # Let's say we requested one frame for the scaling tab.
        # We need to know which frame we got. The dictionary keys help here.
        if not result:
            log.warning("Frame loader returned no frames.")
            return

        # Example: Get the first (and likely only) frame from the result.
        it = iter(result)
        first_frame_index = next(it)
        self.scale_image = result[first_frame_index]
        if self.scale_image is None:
            log.error("Did not receive scale frame")
        # THE MOST IMPORTANT STEP: Notify the UI that new data is ready!
        self.notify_observers('scale_image', self.scale_image)
        self.frame_count = next(it)
        log.info(f"Frame count found: {self.frame_count}")
        self.notify_observers('frame_count', self.frame_count)

    def load_left_frame(self, index=None):
        if index is None:
            index = self.trim_start
        if self.video:
            log.info(f"Loading {index} frame.")
            self.left_index = index
            self.task_manager.queue_task(
                frame_loader,  # The function to run
                self.video,  # This will be the 'vid_file' argument
                [self.left_index],  # This will be the 'frame_indices' argument
                on_result=self.left_frame_loaded # Optional: a method in DataPipeline to handle the result
            )

    def left_frame_loaded(self, result: dict):
        first_frame_index = next(iter(result))
        self.left_image = result[first_frame_index]
        self.notify_observers('left_image', self.left_image)
        self.apply_leveling()
        self.apply_thresh()

    def load_right_frame(self, index=None):
        if index is None:
            index = self.trim_start
        if self.video:
            log.info(f"Loading {index} frame.")
            self.right_index = index
            self.task_manager.queue_task(
                frame_loader,  # The function to run
                self.video,  # This will be the 'vid_file' argument
                [self.right_index],  # This will be the 'frame_indices' argument
                on_result=self.right_frame_loaded # Optional: a method in DataPipeline to handle the result
            )

    def right_frame_loaded(self, result: dict):
        first_frame_index = next(iter(result))
        self.right_image = result[first_frame_index]
        self.notify_observers('right_image', self.right_image)
        self.apply_leveling()
        self.apply_thresh()

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
        if self.left_image is None:
            return

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
        left_img = self.left_image.astype(np.float32)
        self.left_leveled = ((left_img - new_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        if self.right_image is not None:
            right_img = self.right_image.astype(np.float32)
            self.right_leveled = ((right_img - new_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        else:
            self.right_leveled = None
        self.notify_observers('leveled', [self.left_leveled, self.right_leveled])
        #self.apply_thresh()

    def apply_thresh(self):
        """
        Apply a binary threshold to the *transformed_image*, not baseline.
        Pixels >= self.threshold become 255; others 0.
        """
        if self.left_leveled is None:
            return

        th = self.threshold  # (for a 2D grayscale image)
        self.left_threshed = np.where(self.left_leveled >= th, 255, 0).astype(np.uint8)

        if self.right_image is not None:
            self.right_threshed = np.where(self.right_leveled >= th, 255, 0).astype(np.uint8)
        else:
            self.right_threshed = None
        self.notify_observers('threshed', [self.left_threshed, self.right_threshed])
        #self.notify_observers('thimage', [self.threshed_image_first, self.threshed_image])

