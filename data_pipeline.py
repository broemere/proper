from PySide6.QtCore import QObject, Signal
from processing.data_transform import zero_data, smooth_data, label_image, create_visual_from_labels, convert_numpy, restore_numpy, n_closest_numbers
from processing.data_loader import load_csv, frame_loader
from collections import OrderedDict
import numpy as np
import logging
import json
import os
from config import APP_VERSION
from pathlib import Path
from scipy.interpolate import make_splrep, sproot
from widgets.error_bus import user_error

log = logging.getLogger(__name__)

class DataPipeline(QObject):
    # --- SIGNALS ---

    state_loaded = Signal()

    # Scale inputs
    known_length_changed = Signal(float)
    pixel_length_changed = Signal(float)
    scale_is_manual_changed = Signal(bool)
    manual_conversion_factor_changed = Signal(float)
    conversion_factor_changed = Signal(float)

    # Frames
    final_pressure_changed = Signal(float)
    left_keypoint_changed = Signal(int)
    right_keypoint_changed = Signal(int)
    left_image_changed = Signal(np.ndarray)
    right_image_changed = Signal(np.ndarray)

    # Levels
    leveled_images = Signal(tuple)
    brightness_changed = Signal(int)
    contrast_changed = Signal(int)

    # Threshold
    threshed_images = Signal(tuple)
    threshold_changed = Signal(int)

    # Area
    visualization_changed = Signal(dict)
    area_data_changed = Signal(str, object)

    # Export
    n_ellipses_changed = Signal(int)
    results_updated = Signal(dict)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_data = {"t": [], "p": []}
        self.trimmed_data = {"t": [], "p": []}
        self.zeroed_data = {"t": [], "p": []}
        self.smoothed_data = {"t": [], "p": []}
        self.csv_path = None
        self.video = None
        self.frame_count = 0
        self.initial_pressure = 0
        self.final_pressure = 25
        self.left_index = 0
        self.right_index = 0
        self.brightness = 50
        self.contrast = 50
        self.threshold = 127
        self.smoothing_method = "Min"
        self.smoothing_window = 100
        self.zeroing_method = "Min"
        self.zeroing_window = 7
        self.trim_start = 0
        self.trim_stop = 42
        self._observers = {}
        self.task_manager = None
        self.VERSION = APP_VERSION

        self.left_image: np.ndarray | None = None
        self.right_image: np.ndarray | None = None
        self.left_threshed: np.ndarray | None = None
        self.right_threshed: np.ndarray | None = None

        self.left_level_blobs: np.ndarray | None = None
        self.right_level_blobs: np.ndarray | None = None
        self.left_thresh_blobs: np.ndarray | None = None
        self.right_thresh_blobs: np.ndarray | None = None

        self.MAX_MARKERS = 5
        self.area_data_left = OrderedDict()  # {blob_id: [area, cx, cy, 'Label']}
        self.area_data_right = OrderedDict()

        self.thickness_data = []

        self.infusion_rate = 0.5  # uL/sec
        self.MMHG2KPA = 1/7.501

        self.pressures_of_interest = [5, 10, 15, 20, 25]

        # SCALE TAB
        self.known_length = 0.0
        self.pixel_length = 0.0
        self.scale_is_manual = False
        self.manual_conversion_factor = 0.0
        self.conversion_factor = 0.0 # The final, authoritative value

        # AREA TAB
        self.left_threshed_old: np.ndarray | None = None
        self.right_threshed_old: np.ndarray | None = None

        # EXPORT TAB
        self.n_ellipses = 0


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
        # The +1 is necessary because Python slicing is exclusive of the stop index.
        if len(self.raw_data["t"]) > 0:
            start_idx = int(self.trim_start)
            stop_idx = int(self.trim_stop) + 1
            # 1. Apply trimming.
            self.trimmed_data = {
                key: self.raw_data[key][start_idx:stop_idx]
                for key in self.raw_data
            }
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
        print(self.csv_path)
        print(os.path.splitext(os.path.basename(self.csv_path))[0])
        self.raw_data = load_csv(file_path)
        #self.set_trimming(0, self.working_length-1)
        self.notify_observers('raw', None)
        log.info("Finding initial keypoints based on default pressures.")
        self.update_pipeline()
        self.find_and_set_keypoint_by_pressure('left', self.initial_pressure)
        self.find_and_set_keypoint_by_pressure('right', self.final_pressure)
        #

    def set_trimming(self, start: int, stop: int):
        """
        Set the trimming parameters and schedule an update.
        """
        # Clamp values to be within the valid range of the working data length
        log.info(("Setting trim:", start, stop))
        valid_max = self.working_length - 1
        self.trim_start = max(0, min(start, valid_max))
        # Ensure stop is never less than start
        self.trim_stop = max(self.trim_start, min(stop, valid_max))
        self.update_pipeline()
        log.info(("TRIM VALUES", self.trim_start, self.trim_stop))
        self.notify_observers('trimming', (self.trim_start, self.trim_stop))

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

    def load_video_file(self, file_path: str, index=None):
        self.video = file_path
        if index is None:
            index = self.trim_start
        self.task_manager.queue_task(
            frame_loader,  # The function to run
            self.video,  # This will be the 'vid_file' argument
            [index],  # This will be the 'frame_indices' argument
            True,
            on_result=self.initial_frame_loaded # Optional: a method in DataPipeline to handle the result
        )

    def initial_frame_loaded(self, result: dict):
        """
        Callback function executed when the frame_loader task is complete.
        'result' is the dictionary of NumPy arrays returned by frame_loader.
        """
        log.info(f"Received {len(result)-1} loaded frames from worker.")
        if not result:
            log.warning("Frame loader returned no frames.")
            return
        it = iter(result)
        first_frame_index = next(it)
        self.left_image = result[first_frame_index]
        if self.left_image is None:
            log.error("Did not receive left frame")
        self.left_image_changed.emit(self.left_image)
        self.frame_count = next(it)
        log.info(f"Frame count found: {self.frame_count}")
        self.set_trimming(0, self.working_length -1)
        if self.left_image is not None and self.right_image is None:
            self.find_and_set_keypoint_by_pressure('right', self.final_pressure)
        self.level_update()

    @property
    def working_length(self) -> int:
        """
        Returns the effective number of samples for analysis, which is the
        minimum of the CSV data length and the video frame count.
        Returns 0 if either data source is not loaded.
        """
        if not self.raw_data["t"] and self.frame_count == 0:
            return 0

        if self.frame_count == 0:
            return len(self.raw_data["t"])
        elif not self.raw_data["t"]:
            return self.frame_count
        else:
            return min(len(self.raw_data["t"]), self.frame_count)

    def get_state(self, debug_json=False):
        """
        Collects all serializable attributes into a dictionary for saving.

        This method handles the conversion of non-serializable objects (like QPointF)
        into a format that can be pickled.
        """
        # Start with a dictionary of all of the object's attributes
        state = vars(self).copy()

        # --- Data Conversion and Exclusion ---
        # 1. Exclude any attributes that cannot or should not be saved.
        state.pop('task_manager', None)
        state.pop('_observers', None)
        state.pop('left_leveled', None)
        state.pop('right_leveled', None)
        state.pop('left_threshed', None)
        state.pop('right_threshed', None)
        state.pop('left_threshed_old', None)
        state.pop('right_threshed_old', None)

        # 2. Convert any non-serializable objects into a savable format.
        # Here, we convert the list of QPointF objects into a list of tuples.
        #if 'key_locations' in state and state['key_locations']:
            # This list comprehension creates a new list of simple (x, y) tuples
        #    state['key_locations'] = [(p.x, p.y) for p in state['key_locations']]

        state = convert_numpy(state)

        if not debug_json:
            print("\n--- Running JSON Serialization Check ---")
            for key, value in state.items():
                try:
                    # this will raise TypeError (or OverflowError) if not serializable
                    json.dumps(value)
                except (TypeError, OverflowError) as e:
                    print(f"[JSON SERIALIZATION FAILED] key='{key}':")
                    print(f"  • Type : {type(value).__name__}")
                    print(f"  • Value: {value!r}")
                    print(f"  • Error: {e}")
            print("--- Debug Check Complete ---\n")

        log.info(f"Analysis state compiled")
        self.notify_observers('state_dict', state)

        print("\n--- JSON Size Analysis (per key) ---")
        sizes = {}
        for key, value in state.items():
            try:
                j = json.dumps(value)
                size_bytes = len(j.encode('utf-8'))
                sizes[key] = size_bytes
            except Exception:
                sizes[key] = None
        # Sort descending by size (None last)
        for key, sz in sorted(sizes.items(),
                              key=lambda kv: (kv[1] is None, kv[1]),
                              reverse=True):
            if sz is None:
                print(f"{key:30s}: <failed to serialize>")
            else:
                print(f"{key:30s}: {sz/1e6:8.3f} MB")
        print("--- Size Analysis Complete ---\n")

        return state

    def set_state(self, state_dict):
        """
        Restores the object's state from a dictionary (loaded from a file).

        This method handles the reverse conversion of data back into its
        original object representation (e.g., tuples back to QPointF).
        """
        # First, convert data back to its special object format
        #if 'key_locations' in state_dict and state_dict['key_locations']:
        #    state_dict['key_locations'] = [QPointF(x, y) for x, y in state_dict['key_locations']]

        # Update the object's attributes with the values from the dictionary
        for key, value in state_dict.items():
            setattr(self, key, value)

        print("\n--- Pipeline state has been restored ---")

    def load_session(self, state_dict):

        print("Loaded session!")
        fixed_dict = restore_numpy(state_dict)
        print("Setting variables...")
        for k, v in fixed_dict.items():
            setattr(self, k, v)
            print(k, v)
        print("Refreshing...")
        self.refresh_session()

    def refresh_session(self):
        self.update_pipeline()
        print("Plot data restored")
        self.notify_observers('conversion_factor', self.conversion_factor)
        self.notify_observers('author', self.author)
        #self.notify_observers('left_image', self.left_image) # Scale, frame
        self.left_image_changed.emit(self.left_image)
        self.notify_observers('right_image', self.right_image) # Frame, Thickness
        self.level_update()
        self.segment_image(self.left_threshed, "left")
        self.segment_image(self.right_threshed, "right")
        self.left_threshed_old = self.left_threshed.copy()
        self.right_threshed_old = self.right_threshed.copy()
        self.notify_observers('state_loaded', "") # Final notification that the pipeline has had its state reloaded

    def on_author_changed(self, new_author):
        self.author = new_author
        log.info(f"User: {self.author}")

    def set_thickness_data(self, lengths: list[float]):
        """
        Replaces the current thickness data with a new, complete list of line lengths.
        This single method handles adding, undoing, and clearing lines.
        """
        self.thickness_data = lengths
        log.info(f"Pipeline thickness data updated. {len(lengths)} lines.")
        # Notify any observers (like ThicknessTab) that the state has changed.
        self.notify_observers('thickness_data_updated', lengths)

    def pad_array(self, arr, target_len):
        """Pads a short array to a target length with empty strings."""
        # Create a new array of the target length, filled with empty strings
        # and using the 'object' dtype to allow for mixed types.
        padded_arr = np.full(target_len, np.nan, dtype=float)

        # Copy the original data into the beginning of the new array
        padded_arr[:len(arr)] = arr

        return padded_arr

    def get_stress_stretch(self):

        n = len(self.area_data_left)
        if n < 5:
            raise UserError(
                "Area analysis is incomplete.",
                hint=f"You selected {n}/5 blobs in first image. Please click all 5, then retry."
            )

        n = len(self.area_data_right)
        if n < 5:
            raise UserError(
                "Area analysis is incomplete.",
                hint=f"You selected {n}/5 blobs in last image. Please click all 5, then retry."
            )


        data = self.smoothed_data
        t_final = np.mean(n_closest_numbers(self.thickness_data, max(1, len(self.thickness_data)-1))) / self.conversion_factor
        blobs_left = self.area_data_left
        blobs_right = self.area_data_right

        areas_left = []
        areas_right = []

        for props in blobs_left.values():
            area, cx, cy, label = props
            if label == "Middle":
                area_mid_left = area / (self.conversion_factor ** 2)
                ra_left = np.sqrt(area_mid_left/np.pi)
            else:
                areas_left.append(area / (self.conversion_factor ** 2))

        area_left = np.mean(n_closest_numbers(areas_left, self.n_ellipses))
        rb_left = (area_left/(np.pi*ra_left))

        for props in blobs_right.values():
            area, cx, cy, label = props
            if label == "Middle":
                area_mid_right = area / (self.conversion_factor ** 2)
                ra_right = np.sqrt(area_mid_right/np.pi)
            else:
                areas_right.append(area / (self.conversion_factor ** 2))

        area_right = np.mean(n_closest_numbers(areas_right, self.n_ellipses))
        rb_right = (area_right/(np.pi*ra_right))

        v_ext_left = (4 / 3) * np.pi * (ra_left * ra_left * rb_left)
        v_ext_right = (4 / 3) * np.pi * (ra_right * ra_right * rb_right)

        v_int_right = (4 / 3) * np.pi * ((ra_right-t_final) * (ra_right-t_final) * (rb_right-t_final))
        v_wall = v_ext_right - v_int_right
        v_int_left = v_ext_left - v_wall

        coeffs = [1, (-2*ra_left-rb_left), (ra_left**2+2*ra_left*rb_left), -(ra_left**2)*rb_left+(3/4)*(1/np.pi)*(v_ext_left-v_wall)]
        roots = np.roots(coeffs)
        for r in roots:
            if not np.iscomplex(r):
                t_left = float(r)

        frames = np.arange(self.left_index, self.right_index+1)
        start = self.left_index - self.trim_start
        stop = self.right_index - self.trim_start + 1
        p_smoothed = self.smoothed_data["p"][start:stop]
        thickness = np.linspace(t_left, t_final, len(frames))  # Assume linear
        diameter = 2*((np.linspace(v_int_left, v_int_right, len(frames))*(3/(4*np.pi)))**(1/3)) + thickness  # Assume linear
        initial_diameter = diameter[0]
        stretch = diameter / initial_diameter
        stress = (np.array(p_smoothed)*self.MMHG2KPA)*(diameter/2)/(2*thickness)

        self.stress = stress
        self.stretch = stretch

    def generate_report(self):
        print("Indices:", self.left_index, self.right_index)

        data = self.smoothed_data
        t_final = np.mean(n_closest_numbers(self.thickness_data, max(1, len(self.thickness_data)-1))) / self.conversion_factor
        blobs_left = self.area_data_left
        blobs_right = self.area_data_right
        #print(t_final)
        print(blobs_left)
        print(blobs_right)

        areas_left = []
        areas_right = []

        for props in blobs_left.values():
            area, cx, cy, label = props
            if label == "Middle":
                area_mid_left = area / (self.conversion_factor ** 2)
                ra_left = np.sqrt(area_mid_left/np.pi)
            else:
                areas_left.append(area / (self.conversion_factor ** 2))

        area_left = np.mean(n_closest_numbers(areas_left, self.n_ellipses))
        rb_left = (area_left/(np.pi*ra_left))

        for props in blobs_right.values():
            area, cx, cy, label = props
            if label == "Middle":
                area_mid_right = area / (self.conversion_factor ** 2)
                ra_right = np.sqrt(area_mid_right/np.pi)
            else:
                areas_right.append(area / (self.conversion_factor ** 2))

        area_right = np.mean(n_closest_numbers(areas_right, self.n_ellipses))
        rb_right = (area_right/(np.pi*ra_right))

        v_ext_left = (4 / 3) * np.pi * (ra_left * ra_left * rb_left)
        v_ext_right = (4 / 3) * np.pi * (ra_right * ra_right * rb_right)

        v_int_right = (4 / 3) * np.pi * ((ra_right-t_final) * (ra_right-t_final) * (rb_right-t_final))
        v_wall = v_ext_right - v_int_right
        v_int_left = v_ext_left - v_wall

        print("Radius left", ra_left, rb_left)
        print("Radius right", ra_right, rb_right)
        print("Mid area", area_mid_left, area_mid_right)
        print("Areas", area_left, area_right)
        print("V_ext", v_ext_left, v_ext_right)
        print("V_wall", v_wall)
        print("V_int", v_int_left, v_int_right)


        coeffs = [1, (-2*ra_left-rb_left), (ra_left**2+2*ra_left*rb_left), -(ra_left**2)*rb_left+(3/4)*(1/np.pi)*(v_ext_left-v_wall)]
        roots = np.roots(coeffs)
        for r in roots:
            if not np.iscomplex(r):
                t_left = float(r)

        print("Thickness", t_left, t_final)

        frames = np.arange(self.left_index, self.right_index+1)

        start = self.left_index - self.trim_start
        stop = self.right_index - self.trim_start + 1

        print(start, stop)

        t_trimmed = self.trimmed_data["t"][start:stop]
        p_trimmed = self.trimmed_data["p"][start:stop]
        p_zeroed = self.zeroed_data["p"][start:stop]
        p_smoothed = self.smoothed_data["p"][start:stop]
        #v_infused = np.array(t_trimmed) * self.infusion_rate
        #v_resid = v_int_right - np.max(v_infused)
        #v_corrected = v_infused + v_resid

        thickness = np.linspace(t_left, t_final, len(frames))  # Assume linear

        # volume = np.linspace(v_ext_left, v_ext_right, len(frames))
        # ra = np.linspace(ra_left, ra_right, len(frames))
        # rb = np.linspace(rb_left, rb_right, len(frames))
        # thickness_corrected = []
        # for i, v in enumerate(volume):
        #     coeffs = [1, (-2 * ra[i] - rb[i]), (ra[i] ** 2 + 2 * ra[i] * rb[i]),
        #               -(ra[i] ** 2) * rb[i] + (3 / 4) * (1 / np.pi) * (v - v_wall)]
        #     roots = np.roots(coeffs)
        #     for r in roots:
        #         if not np.iscomplex(r):
        #             thickness_corrected.append(float(r))  # Goes negative


        diameter = 2*((np.linspace(v_int_left, v_int_right, len(frames))*(3/(4*np.pi)))**(1/3)) + thickness  # Assume linear

        volume = np.linspace(v_int_left, v_int_right, len(frames))  # Assume linear

        #diameter = 2*((((v_infused + v_int_left)*3)/(4*np.pi))**(1/3))
        initial_diameter = diameter[0]
        stretch = diameter / initial_diameter
        stress = (np.array(p_smoothed)*self.MMHG2KPA)*(diameter/2)/(2*thickness)

        deriv_spline = self.p_spline.derivative()


        stiffnesses = {}
        for p_target in self.pressures_of_interest:
            # 3. Find the time 't' (frame index) where p(t) = p_target
            i = np.argmin(np.abs(np.array(p_zeroed) - p_target))
            stretch_target = stretch[i]
            modulus = deriv_spline(stretch_target)
            stiffnesses[str(p_target)] = {'modulus_kPa': modulus, 'stretch': stretch_target}

        data = np.column_stack([
            frames,
            t_trimmed,
            p_trimmed,
            p_zeroed,
            p_smoothed,
            thickness,
            diameter,
            volume,
            stretch,
            stress,
            self.pad_array(self.pressures_of_interest, len(frames)),
            self.pad_array([stiffnesses[str(p)]["modulus_kPa"] for p in self.pressures_of_interest], len(frames)),
            self.pad_array([stiffnesses[str(p)]["stretch"] for p in self.pressures_of_interest], len(frames)),
        ])

        # Optional: replace inf with nan so Excel doesn't choke
        data = np.where(np.isfinite(data), data, np.nan)

        # Prepare header
        header = "frame,t_trimmed,p_trimmed,p_zeroed,p_smoothed,thickness,diameter(midwall),v_inner,stretch,stress,pressures of interest,stiffness(kpa),stretch"

        # Ensure parent dir exists

        filename = os.path.splitext(os.path.basename(self.csv_path))[0]
        if filename.endswith("_pressure"):
            filename = filename[:-9]

        folder = os.path.dirname(self.csv_path)

        filepath = Path(f"{folder}/{filename}_results.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(filepath):
            print("Results already found")
            i = 2
            filepath = Path(f"{folder}/{filepath.stem}_{i}{filepath.suffix}")
            print(filepath)
            while os.path.exists(filepath):
                print("Many results already exist")
                i += 1
                filepath = Path(f"{folder}/{filepath.stem[:-2]}_{i}{filepath.suffix}")
                print(filepath)

        # Save CSV (Excel-friendly)
        # fmt="%.10g" keeps numbers compact but precise; tweak if you want more/less precision
        np.savetxt(
            filepath,
            data,
            delimiter=",",
            header=header,
            comments="",  # avoid '# ' prefix in header
            fmt="%.10g",
        )

        results_output = {"first": {
                                    "Pressure": self.initial_pressure,
                                    "Frame/Row": self.left_index,
                                    "Radius a": ra_left,
                                    "Radius b": rb_left,
                                    "Wall Thickness": t_left,
                                    "Volume": v_ext_left,
                                    "V_wall": v_wall,
                                    "V_lumen": v_int_left,
        },
            "last": {
                "Pressure": self.final_pressure,
                "Frame/Row": self.right_index,
                "Radius a": ra_right,
                "Radius b": rb_right,
                "Wall Thickness": t_final,
                "Volume": v_ext_right,
                "V_wall": v_wall,
                "V_lumen": v_int_right,
            }
        }

        self.results_updated.emit(results_output)


    ### SCALE TAB

    def _recalculate_conversion_factor(self):
        """Central calculation. Called whenever an input changes."""
        new_factor = 0.0
        if self.scale_is_manual:
            new_factor = self.manual_conversion_factor
        elif self.known_length > 0 and self.pixel_length > 0:
            new_factor = self.pixel_length / self.known_length

        # Use the main setter to update the value and emit the signal
        self.set_conversion_factor(new_factor, force_update=True)

    def set_known_length(self, length: float):
        if self.known_length != length:
            self.known_length = length
            self.known_length_changed.emit(self.known_length)
            self._recalculate_conversion_factor()

    def set_pixel_length(self, length: float):
        if self.pixel_length != length:
            self.pixel_length = length
            self.pixel_length_changed.emit(self.pixel_length)
            self._recalculate_conversion_factor()

    def set_scale_is_manual(self, is_manual: bool):
        if self.scale_is_manual != is_manual:
            self.scale_is_manual = is_manual
            self.scale_is_manual_changed.emit(self.scale_is_manual)
            self._recalculate_conversion_factor()

    def set_manual_conversion_factor(self, factor: float):
        """This is called when the user types in the manual spinbox."""
        if self.manual_conversion_factor != factor:
            self.manual_conversion_factor = factor
            # Only recalculate if we are currently in manual mode
            self.manual_conversion_factor_changed.emit(self.manual_conversion_factor)
            if self.scale_is_manual:
                self._recalculate_conversion_factor()

    def set_conversion_factor(self, factor: float, force_update=False):
        """
        This is the final setter for the authoritative value.
        It's called by the recalculate method or can be set directly.
        """
        if self.conversion_factor != factor or force_update:
            self.conversion_factor = factor
            log.info(f"Conversion factor updated: {self.conversion_factor}")
            self.notify_observers('conversion_factor', factor)
            self.conversion_factor_changed.emit(self.conversion_factor)


    ### FRAME TAB

    def set_final_pressure(self, pressure: float):
        """Sets the final pressure and emits a signal if it changes."""
        if self.final_pressure != pressure:
            self.final_pressure = pressure
            self.final_pressure_changed.emit(self.final_pressure)

    def set_left_keypoint(self, index: int, load_frame: bool = True):
        """
        Sets the left keypoint index, validates it, and optionally loads the frame.
        """
        # Validate and clamp the index to be within the current trim range
        self.left_index = max(self.trim_start, min(index, self.trim_stop))
        log.info(f"Setting left keypoint index to: {self.left_index}")
        if load_frame and self.video:
            log.info(f"Dispatching frame loader for left_index: {self.left_index}")
            self.task_manager.queue_task(
                frame_loader, self.video, [self.left_index], on_result=self.left_frame_loaded
            )
        self.left_keypoint_changed.emit(self.right_index)

    def set_right_keypoint(self, index: int, load_frame: bool = True):
        """
        Sets the right keypoint index, validates it, and optionally loads the frame.
        """
        self.right_index = max(self.trim_start, min(index, self.trim_stop))
        log.info(f"Setting right keypoint index to: {self.right_index}")
        if load_frame and self.video:
            log.info(f"Dispatching frame loader for right_index: {self.right_index}")
            self.task_manager.queue_task(
                frame_loader, self.video, [self.right_index], on_result=self.right_frame_loaded
            )
        self.right_keypoint_changed.emit(self.right_index)

    def left_frame_loaded(self, result: dict):
        first_frame_index = next(iter(result))
        self.left_image = result[first_frame_index]
        self.left_image_user = None
        self.left_image_changed.emit(self.left_image)
        self.level_update()

    def right_frame_loaded(self, result: dict):
        first_frame_index = next(iter(result))
        self.right_image = result[first_frame_index]
        self.right_image_user = None
        self.right_image_changed.emit(self.right_image)
        self.level_update()

    def _find_closest_index_by_pressure(self, target_pressure: float) -> int:
        """
        Finds the index in the smoothed data closest to a target pressure.
        Returns:
            int: The absolute index (relative to raw_data) of the closest point.
        """
        if self.smoothed_data['p'].size == 0:
            log.warning("Cannot find pressure; smoothed data is empty.")
            return self.trim_start  # Return a safe default
        # Use numpy for efficient searching
        pressure_array = np.asarray(self.smoothed_data['p'])
        # Find the index in the *trimmed* data array that has the minimum difference
        relative_index = np.argmin(np.abs(pressure_array - target_pressure))
        # Convert the relative index back to an absolute index by adding the trim_start offset
        absolute_index = self.trim_start + relative_index
        return int(absolute_index)

    def find_and_set_keypoint_by_pressure(self, side: str, pressure: float):
        """
        Public method for the UI to find a keypoint by its pressure value.
        Args:
            side (str): Either 'left' or 'right'.
            pressure (float): The target pressure to find.
        """
        log.info(f"UI requested to find keypoint for '{side}' side at {pressure} mmHg.")
        # First, store the user's requested pressure so it's saved in the state
        if side == 'left':
            self.initial_pressure = pressure
        elif side == 'right':
            self.final_pressure = pressure  # Possible duplicate with set_final_pressure
        # Find the absolute index corresponding to that pressure
        found_index = self._find_closest_index_by_pressure(pressure)
        # Use our existing, validated setters to update the state and load the frame
        if side == 'left':
            self.set_left_keypoint(found_index)
        elif side == 'right':
            self.set_right_keypoint(found_index)

    def get_pressure_display_data(self, index: int) -> dict:
        """
        Processes and returns the pressure data needed for the UI display.
        The Model is responsible for this logic.
        """
        if not self.csv_path or "p" not in self.smoothed_data:
            return {"pre": "", "current": "0.00", "post": ""}
        # All the slicing and formatting logic is now here
        new_index = index - self.trim_start
        p_data = self.smoothed_data["p"]
        pre_vals = p_data[max(0, new_index - 3):new_index]
        current_val = p_data[new_index]
        post_vals = p_data[new_index+1:new_index+4]
        return {
            "pre": ", ".join(f"{x:.2f}" for x in pre_vals),
            "current": f"{current_val:.2f}",
            "post": ", ".join(f"{x:.2f}" for x in post_vals)
        }


    ### LEVEL TAB

    def set_brightness(self, value: int):
        """Sets the brightness level, emits a signal, and triggers an update."""
        if self.brightness != value:
            self.brightness = value
            self.brightness_changed.emit(self.brightness)
            self.level_update()

    def set_contrast(self, value: int):
        """Sets the contrast level, emits a signal, and triggers an update."""
        if self.contrast != value:
            self.contrast = value
            self.contrast_changed.emit(self.contrast)
            self.level_update()

    def reset_levels(self):
        """Resets all level-related parameters to their defaults."""
        self.set_brightness(50)
        self.set_contrast(50)
        self.left_level_blobs = None
        self.right_level_blobs = None
        self.level_update()

    def update_level_blobs(self, side: str, image_array: np.ndarray):
        """Handles the logic of merging a new drawing into the existing blobs."""
        if side == "left":
            if self.left_level_blobs is not None:
                mask = (image_array != 127)  # Logic is now inside the model
                self.left_level_blobs[mask] = image_array[mask]
            else:
                self.left_level_blobs = image_array.astype(np.uint8)
        elif side == "right":
            if self.right_level_blobs is not None:
                mask = (image_array != 127)  # Logic is now inside the model
                self.right_level_blobs[mask] = image_array[mask]
            else:
                self.right_level_blobs = image_array.astype(np.uint8)
        self.level_update()

    def level_update(self):
        frames = self.apply_leveling()
        self.apply_thresh(frames)
        self.left_leveled, self.right_leveled = frames

    def paste_level_blobs(self, left_img, right_img):
        if self.left_level_blobs is not None:
            white = np.max(left_img)
            black = np.min(left_img)
            mask = (self.left_level_blobs != 127)
            clamped = np.clip(self.left_level_blobs, black, white)
            left_img[mask] = clamped[mask]
        if self.right_level_blobs is not None:
            white = np.max(right_img)
            black = np.min(right_img)
            mask = (self.right_level_blobs != 127)
            clamped = np.clip(self.right_level_blobs, black, white)
            right_img[mask] = clamped[mask]
        return left_img, right_img

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
        left_leveled = ((left_img - new_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        if self.right_image is not None:
            right_img = self.right_image.astype(np.float32)
            right_leveled = ((right_img - new_min) / denom * 255.0).clip(0, 255).astype(np.uint8)
        else:
            right_leveled = None

        left_leveled, right_leveled = self.paste_level_blobs(left_leveled, right_leveled)

        self.leveled_images.emit((left_leveled, right_leveled))
        return [left_leveled, right_leveled]


    ### THRESHOLD TAB

    def set_threshold(self, value: int):
        """Sets the brightness level, emits a signal, and triggers an update."""
        if self.threshold != value:
            self.threshold = value
            self.threshold_changed.emit(self.threshold)
            self.apply_thresh()

    def reset_thresh(self):
        """Resets all level-related parameters to their defaults."""
        self.set_threshold(127)
        self.left_thresh_blobs = None
        self.right_thresh_blobs = None
        self.apply_thresh()

    def update_thresh_blobs(self, side: str, image_array: np.ndarray):
        """Handles the logic of merging a new drawing into the existing blobs."""
        if side == "left":
            if self.left_thresh_blobs is not None:
                mask = (image_array != 127)  # Logic is now inside the model
                self.left_thresh_blobs[mask] = image_array[mask]
            else:
                self.left_thresh_blobs = image_array.astype(np.uint8)
            self.left_threshed_old = self.left_threshed.copy()
            self.segment_image(self.left_threshed, "left")
        elif side == "right":
            if self.right_thresh_blobs is not None:
                mask = (image_array != 127)  # Logic is now inside the model
                self.right_thresh_blobs[mask] = image_array[mask]
            else:
                self.right_thresh_blobs = image_array.astype(np.uint8)
            self.right_threshed_old = self.right_threshed.copy()
            self.segment_image(self.right_threshed, "right")
        self.apply_thresh()

    def paste_thresh_blobs(self, left_img, right_img):
        if self.left_thresh_blobs is not None:
            mask = (self.left_thresh_blobs != 127)
            left_img[mask] = self.left_thresh_blobs[mask]
        if self.right_thresh_blobs is not None:
            mask = (self.right_thresh_blobs != 127)
            right_img[mask] = self.right_thresh_blobs[mask]
        return left_img, right_img

    def apply_thresh(self, frame_data=None):
        """
        Apply a binary threshold to the *transformed_image*, not baseline.
        Pixels >= self.threshold become 255; others 0.
        """
        if frame_data is None:
            left_leveled = self.left_leveled
            right_leveled = self.right_leveled
        else:
            left_leveled, right_leveled = frame_data

        if left_leveled is None:
            return

        th = self.threshold  # (for a 2D grayscale image)
        self.left_threshed = np.where(left_leveled >= th, 255, 0).astype(np.uint8)

        if self.right_image is not None:
            self.right_threshed = np.where(right_leveled >= th, 255, 0).astype(np.uint8)
        else:
            self.right_threshed = None

        self.left_threshed, self.right_threshed = self.paste_thresh_blobs(self.left_threshed, self.right_threshed)
        self.threshed_images.emit((self.left_threshed, self.right_threshed))


    ### AREA TAB

    def segment_image(self, arr, left_right):
        log.info(f"Queueing segmentation ({left_right}) task.")
        self.task_manager.queue_task(
            label_image,  # The function to run
            arr,
            left_right,
            on_result=self.image_segmented # Optional: a method in DataPipeline to handle the result
        )

    def image_segmented(self, result):
        """Handles the result from the label worker and starts the visualization task."""
        if result is not None:
            labels, left_right = result
            log.info("Received segmentation. Queueing visualization task.")
            self.task_manager.queue_task(
                create_visual_from_labels,
                labels,
                left_right, # Pass the labels array to the worker
                on_result=self._visualization_created
            )
        else:
            log.warning("Received None from label worker, aborting visualization.")

    def _visualization_created(self, result):
        """Handles the result from the visualization worker."""
        if result and result['visual'] is not None:
            log.info("Received color visualization from worker.")
            # Notify the UI with the dictionary containing both arrays
            self.visualization_changed.emit(result)
        else:
            log.warning("Visualization worker failed or returned no data.")

    def ensure_segmentation_is_up_to_date(self):
        """
        Checks if the thresholded images have changed and runs segmentation
        if they have. This logic now lives in the model.
        """
        # --- Check Left Image ---
        if self.left_threshed is not None:
            should_run_left = (self.left_threshed_old is None or
                               not np.array_equal(self.left_threshed, self.left_threshed_old))
            if should_run_left:
                self.left_threshed_old = self.left_threshed.copy()
                self.segment_image(self.left_threshed, "left")
        # --- Check Right Image ---
        if self.right_threshed is not None:
            should_run_right = (self.right_threshed_old is None or
                                not np.array_equal(self.right_threshed, self.right_threshed_old))
            if should_run_right:
                self.right_threshed_old = self.right_threshed.copy()
                self.segment_image(self.right_threshed, "right")

    def get_display_area_data(self, side: str) -> list[tuple[str, str]]:
        """
        Returns a list of tuples (position_label, area_string)
        with the area already scaled and formatted.
        """
        data_store = self.area_data_left if side == 'left' else self.area_data_right
        display_data = []

        for props in data_store.values():
            area, cx, cy, label = props
            try:
                scaled_area = area / (self.conversion_factor ** 2)
                area_str = f"{scaled_area:.3f}"
            except (ZeroDivisionError, TypeError):
                area_str = f"{area} (px²)"  # Provide a fallback display value

            display_data.append((label, area_str))
        return display_data

    def add_area_data(self, left_right: str, blob_props: list):
        """
        Adds blob data using an OrderedDict to enforce a FIFO size limit.
        """
        blob_id, area, cx, cy = blob_props
        data_store = self.area_data_left if left_right == 'left' else self.area_data_right
        # If the user re-clicks an existing blob, move it to the end (making it the newest)
        if blob_id in data_store:
            data_store.move_to_end(blob_id)

        data_store[blob_id] = [area, cx, cy, ""]  # Add or update the blob data
        # If the dictionary is now over the max size, pop the oldest item.
        if len(data_store) > self.MAX_MARKERS:
            # .popitem(last=False) removes the first item inserted (FIFO)
            data_store.popitem(last=False)

        if len(data_store) == self.MAX_MARKERS:  # Recalculate labels if we have a full set of 5
            self._calculate_and_assign_labels(data_store)

        self.area_data_changed.emit(left_right, dict(data_store))

    def _calculate_and_assign_labels(self, data_store: dict):
        """
        Calculates and assigns positional labels sequentially to prevent overwrites
        and handle "corner cases" correctly.
        """
        log.info(f"Calculating and assigning labels: {data_store}")
        if len(data_store) != self.MAX_MARKERS:
            return

        # First, clear all previous labels to ensure a clean slate
        for blob_id in data_store:
            data_store[blob_id][3] = ""

        # Create a mutable list of unlabeled blobs with their properties
        # Format: [(blob_id, cx, cy), ...]
        unlabeled_blobs = [
            (blob_id, props[1], props[2]) for blob_id, props in data_store.items()
        ]

        # --- Find, label, and remove blobs one by one ---

        # Find Left-most blob from all candidates
        left_blob = min(unlabeled_blobs, key=lambda b: b[1])
        data_store[left_blob[0]][3] = "Left"
        unlabeled_blobs.remove(left_blob)

        # Find Right-most blob from the *remaining* candidates
        right_blob = max(unlabeled_blobs, key=lambda b: b[1])
        data_store[right_blob[0]][3] = "Right"
        unlabeled_blobs.remove(right_blob)

        # Find Top-most blob from the *remaining* candidates
        top_blob = min(unlabeled_blobs, key=lambda b: b[2])
        data_store[top_blob[0]][3] = "Top"
        unlabeled_blobs.remove(top_blob)

        # Find Bottom-most blob from the *remaining* candidates
        bottom_blob = max(unlabeled_blobs, key=lambda b: b[2])
        data_store[bottom_blob[0]][3] = "Bottom"
        unlabeled_blobs.remove(bottom_blob)

        # The last remaining blob must be the Middle one
        if unlabeled_blobs:
            middle_blob_id = unlabeled_blobs[0][0]
            data_store[middle_blob_id][3] = "Middle"


    ### EXPORT TAB

    def set_n_ellipses(self, value: int):
        if self.n_ellipses != value:
            self.n_ellipses = value
            log.info(f"Pipeline n_ellipses set to: {self.n_ellipses}")
            self.n_ellipses_changed.emit(self.n_ellipses)

