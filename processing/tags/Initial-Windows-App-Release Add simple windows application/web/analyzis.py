import numpy as np

from lib.sdfind.sdfind import sdfind
from lib.glfind.glfind import glfind

from models.models import GenLibAnalyzeError, SizeStandardAnalyzePeaks, SizeStandardAnalyzeError, SizeStandardCalibration, SizeStandardRawSignal, GenLibRawSignal, SizeStandardAnalyzeResult, GenLibAnalyzeResult


def analyze_size_standard(raw_signal: SizeStandardRawSignal, calibration: SizeStandardCalibration) -> SizeStandardAnalyzeResult | SizeStandardAnalyzeError:
    try:
        result = sdfind(
            np.array(raw_signal, dtype=np.float64),
            np.array(calibration.sizes, dtype=np.float64),
            np.array(calibration.release_times, dtype=np.float64),
            np.array(calibration.concentrations, dtype=np.float64),
        )
    except Exception as ex:
        return SizeStandardAnalyzeError(
            state='error',
            message=str(ex),
        )

    return SizeStandardAnalyzeResult(
        state='success',
        peaks=SizeStandardAnalyzePeaks(
            data=result.peaks.tolist(),
            sizes=calibration.sizes,
            concentrations=calibration.concentrations,
        ),
        led_area=result.peak_areas.tolist(),
        led_conc=result.concentrations.tolist(),
        ZrRef=result.corrected_signal.tolist(),
        SD_molarity=result.molarity.tolist(),
        liz_fit=result.size_fit.tolist(),
        locs_fit=result.peak_fit.tolist(),
    )


def analyze_gen_lib(raw_signal: GenLibRawSignal, size_standard_analyze_peaks: SizeStandardAnalyzePeaks) -> GenLibAnalyzeResult | GenLibAnalyzeError:
    try:
        res = glfind(
            np.array(raw_signal, dtype=np.float64),
            np.array(size_standard_analyze_peaks.data, dtype=np.int64),
            np.array(size_standard_analyze_peaks.sizes, dtype=np.float64),
            np.array(size_standard_analyze_peaks.concentrations, dtype=np.float64),
        )
    except Exception as ex:
        return GenLibAnalyzeError(
            state='error',
            message=str(ex),
        )
    return GenLibAnalyzeResult(
        state='success',
        t_main=res.t_main.tolist(),
        denoised_data=res.corrected_data.tolist(),
        st_peaks=res.st_peaks.tolist(),
        st_length=res.st_length.tolist(),
        t_unrecognized_peaks=res.t_unrecognized_peaks.tolist(),
        unrecognized_peaks=res.unrecognized_peaks.tolist(),
        lib_length=res.lib_length.tolist(),
        LibPeakLocations=res.lib_peak_locations.tolist(),
        t_final_locations=res.t_final_locations.tolist(),
        final_filtered_below_threshold_locations=res.final_lib_local_minimums.tolist(),
        hpx=res.hpx.tolist(),
        unr=res.unr.tolist(),
        stp=res.stp.tolist(),
        mainCorr=res.main_corr.tolist(),

        GLAreas=res.all_areas.tolist(),
        peaksCorr=res.all_peaks_corr.tolist(),
        library_peaks=res.all_peaks.tolist(),
        areaCorr=res.all_areas_conc.tolist(),
        molarity=res.molarity.tolist(),

        maxLibPeak=float(res.max_lib_peak),
        maxLibValue=float(res.max_lib_value),
        totalLibArea=float(res.total_lib_area),
        totalLibConc=float(res.total_lib_conc),
        totalLibMolarity=float(res.total_lib_molarity),

        x_fill=res.x_fill.tolist(),
        y_fill=res.y_fill.tolist(),
        x_Lib_fill=res.x_lib_fill.tolist(),
        y_Lib_fill=res.y_lib_fill.tolist(),
    )
