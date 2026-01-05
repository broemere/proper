HELP_CONTENT = {
    "smoothing": (
        "Smoothing Tab Help",
        """
        <p><b>Purpose of This Tab</b></p>
        <p>
        This tab applies a filter on the <b>stress-stretch curve</b> for stiffness parameter calculation using
        a mathematical smoothing technique. A properly smoothed curve ensures that the stiffness parameters 
        exported more truly represent the material properties, free from the influence of transient noise.
        </p>

        <p><b>Why Smoothing is Necessary</b></p>
        <p>
        The raw data often contain small fluctuations typically caused by transient pressure events. These 
        events can significantly affect the curve's derivative (i.e., stiffness), especially at high 
        pressures/stress. Smoothing removes this noise to model the underlying trend of the data.
        </p>

        <p><b>How It Works: Spline Smoothing</b></p>
        <p>
        This filter uses a <b>Univariate Spline</b> technique, which 
        creates a set of mathematical curves that pass through or near the data points.
        The result is a new, perfectly continuous curve that represents the 'best fit' of the
        original data, effectively tamping down the wave-like noise.
        </p>

        <p><b>Your Task: Adjust the Smoothing Factor ('s')</b></p>
        <p>
        Your only goal in this tab is to find the ideal smoothing level using the slider.
        The gray line is the original data, and the <b><font color='#E74C3C'>red line</font></b>
        is the smoothed spline.
        </p>
        <ul>
            <li>A <b>low 's' value</b> will cause the spline to follow the noisy data
            very closely <i>(under-smoothing)</i>.</li>

            <li>A <b>high 's' value</b> may cause the spline to become too general,
            ignoring important features of the data <i>(over-smoothing)</i>.</li>

            <li>Adjust the slider to find the <b>'sweet spot'</b> where the red spline
            is perfectly smooth (no small bumps or wiggles) but still faithfully captures
            the overall shape of the original data.</li>
            
            <li>The spline curve itself is NOT exported and NOT used for the final stress-stretch data.</li>
            
            <li>Stiffness is the derivative of the red curve at the pressures 5, 10, 15, 20, and 25 mmHg 
            (shown by black circles). A perfect fit at stress=0/stretch=1 is not necessarily required,
            instead the fit near the circles is more important.
            Sacrificing fit around 0 can improve the quality of the fit on the rest of the curve, 
            and is not detrimental to the final data, since this curve is not exported.</li>
        </ul>

        """
    ),

}

ERROR_CONTENT = {
    "no_conversion_factor": (
        "Issue: Missing conversion factor.",
        "Go to the Scale tab and ensure the conversion factor is set."
    ),
    "area_incomplete": (
        "Issue: Area analysis incomplete.",
        "Go to the Area tab and ensure all 5 blobs have been selected for the "
    ),
    "thickness_incomplete": (
        "Issue: Thickness analysis incomplete.",
        "Go to the Thickness tab and ensure the thickness has been measured at least once."
    ),
    "empty_data_array": (
        "Issue: XY data is missing or invalid.",
        "Go to 'Import CSV' and load a valid data file."
    ),
    "invalid_roots": (
        "Issue: No valid thickness found in backcalculation.",
        "Report to developer."
    ),
    "negative_volume": (
        "Issue: Inner volume found to be negative.",
        "Report to developer."
    )
}