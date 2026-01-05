# PROPER
Python Reconstruction Of PRIM Experiment Recording

A GUI for performing mechanical analyses of PRIM experimental data.

Developed with research scientists. Vastly shortens data analysis time, and reduces human errors.

---

## Features

* Pressure-time data visualization with trimming, smoothing, and zeroing functions
* Pixel-to-mm scaling
* Easy keyframe location (start and end) via pressure (e.g., 0 and 25 mmHg)
* Image brightness, contrast, and thresholding controls
* Robust image painting tools for cleaning up noise and voids
* Automatic area calculation of segmented objects
* User-friendly thickness measuring tool
* Smoothing filter for improved derivative calculations
* Complete data analysis export to csv:
  * Frame number
  * Time
  * Pressure
  * Wall Thickness
  * Mid-wall diameter
  * Inner volume
  * Stretch
  * Stress
  * Stress-stretch average at each 2mmHg pressure intervals
    * Allows for easy comparison across subjects at the same pressures
  * Stiffness calculation at each 5mmHg pressure interval
    * Derivative of forced-continuous smoothing filter at pressure of interest
    * Stress-stretch values nearest to each 5mmHg pressure

---

## Requirements

### Running app

* Windows
* Mac

### Building app

* **Python 3.11+** with packages in `requirements.txt`

---

## Installation

### Direct run

* Download and run the exe/dmg from https://github.com/broemere/proper/releases
* Your OS may block the program the first time it is run.
  `"Windows protected your PC` → `More info` → `Run anyway`

### Building from source code

#### 1. Clone the repository

```bash
git clone https://github.com/broemere/proper.git
```

#### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the app
You can either:

* Run directly from source
    ```bash
    python proper.py
    ```
  
or

* Build a standalone executable
    ```bash
    python build.py
    ```
    Run ```proper_<version>.exe``` in ```/dist```
    * This executable is fully self-contained, can be copied/moved, and no longer requires the source code or build environment.

---

## Usage

1. Run the app.
2. Import your CSV/video file pair. 
3. Verify the pressure-time data **plot** is satisfactory (smoothing, zeroing).
4. Set the pixel/mm **scale**. Zoom on the image for better accuracy. Draw a line on your known length in the image. Enter the numerical value of the known length. 
   * Optionally, there is a tick box to manually enter the scale.
5. Verify your start and end frames by choosing your start/stop pressures. Defaults (0, 25) are loaded automatically.
   * Nearest pressure image is used when exact value is not available.
6. Adjust the image **levels** (brightness, contrast) to your preference.
7. Click `Draw` on each image to hide the neck and extra tissue.
8. Adjust the threshold so there are clear boundaries on each view of the tissue.
9. Click `Draw` on each image to clean up noise (draw with black), and fill in voids (draw with white).
   * Aim is to have solid white objects isolated from the background.
   * It is not necessary to manually remove the background/all noise, just enough to have a clear boundary around your objects.
10. Use the **area** selector and click on each of the 5 objects in BOTH images (first and last).
    * The program will label the objects when you have completed the image.
    * You may go back to the drawing tools to further edit the images at any time if you need to rework the result.
11. Use the **thickness** tool to draw several measures of the wall thickness on your last image.
    * Zoom in a lot for better accuracy (+/- keys, or Ctrl+Scroll Wheel).
    * Click to start drawing a line, click again to finish. Ctrl+Z to undo. Esc cancels a line.
    * Align the bars at the end of the line with the surface.
    * About 10 measures total across multiple views is recommended.
    * The program automatically drops 1 outlier from the measures.
12. Adjust the **smoothing** slider to fit the spline to the stress-stretch curve.
    * The spline should fit the data closely, but be free of bumps/wiggles. 
    * Click `?` for further detailed instructions.
    * This smoothed spline is NOT exported with the final data. It is only used for improved stiffness calculations.
13. **Export results** to a CSV file in the same location as the original data.

---

## Data manipulation

The trade-off of this data analysis tool being user-friendly is that it also obscures the data transformations under the hood. To lift the obscurity, the operations for each column in the `data_results.csv` are summarized below. Items with a ★ are directly controlled by the user. 

* frame: Frame number [-]
  * ★ Defined from the first and last pressures in `Frame Picker`
  * Values always remain true to original video frame number.
* t_trimmed: Time (trimmed) [s]
* p_trimmed: Pressure (trimmed) [mmHg]
  * ★ Defined from trimming tool in `Plot`
  * ★ Further trimmed to match the `Frame Picker` pressure selections.
* p_zeroed: Pressure (zeroed) [mmHg]
  * ★ Defined from zeroing tool in `Plot`. Applied to trimmed data.
  * ★ Options: [None, First, Min, Median] (default: Min)
  * ★ Window: static window size of zeroing operation (default: 7).
  * Finds "zero" value from the window from the beginning of the data. Subtracts that value from the whole data.
  * ★ Further trimmed to match the `Frame Picker` pressure selections.
* p_smoothed: Pressure (smoothed) [mmHg]
  * ★ Defined from smoothing tool in `Plot`. Applied to zeroed data.
  * ★ Options: [None, Minimum, Minimum x2, Moving Average, Median, Gaussian] (default: Min to damp transient pressure events).
    * Min: min(y<sub>i</sub> : y<sub>i+window</sub>)
    * Double min: min(min(y<sub>i</sub> : y<sub>i+window</sub>))
    * Moving Avg: mean(y<sub>i-half_window</sub> : y<sub>i+half_window</sub>)
    * Median: median(y<sub>i-half_window</sub> : y<sub>i+half_window</sub>)
    * **Note** Average and Median are skewed at the ends due to windowing edge effects.
  * ★ Window: moving window size of smoothing operation (default: 100).
  * Hover over the window spinbox to visualize.
  * ★ Further trimmed to match the `Frame Picker` pressure selections.
* thickness: Elliptical Wall Thickness [mm]
  * ★ t_final = mean(thickness measures) (1 outlier automatically dropped).
  * t_initial solved for via equation 7 in DOI 10.1007/s10237-023-01727-0
  * All other ts: assume linear from t_initial → t_final. (Not ideal)
* v_inner: Elliptical Internal volume [mm<sup>3</sup>]
  * ★ Radius A (ra): calculated via middle area = pi × ra<sup>2</sup> for each initial and final images.
  * ★ Radius B (rb): calculated via mean(n side areas) = pi × ra × rb for each initial and final images.
    * ★ n is a user setting in the `Export` tab. Some users prefer to drop 1 outlier area.
  * v_external_final = (4/3) × pi × (ra_final<sup>2</sup> × rb_final)
  * v_internal_final = (4/3) × pi × ((ra_final - t_final)<sup>2</sup> × (rb_final - t_final))
  * v_wall = v_external_final - v_internal_final
  * v_external_initial = (4/3) × pi × (ra_initial<sup>2</sup> × rb_initial)
  * v_internal_initial = v_external_initial - v_wall
  * All other internal vs: assume linear from v_internal_initial → v_internal_final
* diameter: Mid-wall spherical equivalent diameter [mm]
  * d = 2 × ((v_inner × (3/(4 × pi)))<sup>(1/3)</sup>) + thickness
* stretch: Spherical stretch [-]
  * λ = diameter / diameter_initial
* stress: Spherical pressure vessel wall stress [kPa]
  * σ = (p_smoothed / 7.501) × (diameter / 2) / (2 × thickness)
* pressure_intervals: [mmHg]
  * All even pressure integers within the trim range.
* stretch_intervals: Average stretch about the nearest pressure value with a window of 5 [-]
  * λ<sub>i</sub> = mean(stretch[i-2 : i+3])
* stress_intervals: Average stress about the nearest pressure value with a window of 5 [kPa]
  * σ<sub>i</sub> = mean(stress[i-2 : i+3])
* pressures_of_interest: [mmHg]
  * Multiples of 5 within the trim range.
* nearest_pressure: [mmHg]
  * Actual pressure value (p_smoothed) available in the data nearest to the POI.
* stiffness: [kPa]
  * ★ Derivative of spline-smoothed stress-stretch curve at nearest pressure.
* stretch_at_poi: Stretch at pressure of interest [-]
  * stretch(nearest_pressure) (calculated above)
* stress_at_poi: Stress at pressure of interest [kPa]
  * stress(nearest_pressure) (calculated above)
* v_wall: Elliptical wall volume [mm<sup>3</sup>]
  * Calculated above

---

## License

**MIT License** -- see ```LICENSE.md``` for details.

Third-party components are distributed under their own terms, see ```LICENSES/```