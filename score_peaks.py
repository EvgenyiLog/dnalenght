from typing import Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths

def score_peaks_genlib(
    data: Union[np.ndarray, pd.Series, list],
    title: str = "Signal",
    sizes: Optional[Union[np.ndarray, list]] = None
) -> pd.DataFrame:
    """
    Detects peaks in a 1D signal and applies a multi-stage scoring system to rank them.
    
    Parameters
    ----------
    data : np.ndarray | pd.Series | list
        1D array of signal values for peak detection.
    title : str, optional
        Title for the diagnostic plot (shown if no peaks are found). Default is "Signal".
    sizes : np.ndarray | list, optional
        Array of size values to assign to top-ranked peaks. Determines how many peaks to select (n = len(sizes) or 11 if None).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing comprehensive characteristics and scoring results for all detected peaks.
        Columns:
            'Пик №' : Peak index (1-based)
            'Позиция' : Position index in the original data array
            'Высота' : Peak height (amplitude)
            'Ширина FWHM' : Full width at half maximum
            'Площадь' : Area under the peak (trapezoidal integration)
            'Проминенс' : Peak prominence
            'Лев.граница' : Left interpolation position of peak width
            'Прав.граница' : Right interpolation position of peak width
            'Баллы' : Total score from multi-stage evaluation
            'Выбран' : '✓' if peak was selected in top-N, empty string otherwise
    """
    # Ensure data is a numpy array
    data = np.asarray(data, dtype=float)
    
    # Peak detection parameters
    height_threshold = 0.05 * np.max(data)
    distance = max(5, len(data) // 200)
    
    peaks, _ = find_peaks(
        data,
        height=height_threshold,
        distance=distance,
        prominence=0.01 * np.max(data),
        width=1
    )
    
    print(f'\n{"=" * 80}')
    print(f'НАЙДЕНО ПИКОВ: {len(peaks)}')
    print(f'{"=" * 80}\n')
    
    # Handle case with no peaks detected
    if len(peaks) == 0:
        print('Пики не найдены. Попробуйте изменить параметры поиска.')
        
        
        # Return empty DataFrame with expected columns
        columns = [
            'Пик №', 'Позиция', 'Высота', 'Ширина FWHM', 'Площадь', 
            'Проминенс', 'Лев.граница', 'Прав.граница', 'Баллы', 'Выбран'
        ]
        return pd.DataFrame(columns=columns)
    
    # Calculate additional peak characteristics (ensure numpy arrays)
    prominences = np.asarray(peak_prominences(data, peaks)[0])
    widths = np.asarray(peak_widths(data, peaks, rel_height=0.5)[0])
    left_ips = np.asarray(peak_widths(data, peaks, rel_height=0.5)[2])
    right_ips = np.asarray(peak_widths(data, peaks, rel_height=0.5)[3])
    
    # Calculate peak areas using trapezoidal integration
    areas = np.array([
        np.trapz(data[int(left_ips[i]):int(right_ips[i]) + 1])  # trapz for compatibility
        for i in range(len(peaks))
    ])
    
    # Initialize scoring system
    print(f'\n{"=" * 80}')
    print('БАЛЛЬНАЯ ОЦЕНКА ПИКОВ')
    print(f'{"=" * 80}\n')
    
    n_peaks = len(peaks)
    n_sizes = len(sizes) if sizes is not None else 11
    Points = np.zeros(n_peaks)
    
    # === STAGE 1: Area filtering ===
    print('ЭТАП 1: Отбор по площади пиков')
    for i in range(n_peaks):
        if areas[i] > 1:
            Points[i] += 1.0
            print(f'  Пик #{i + 1}: площадь = {areas[i]:.2f} > 1 → +1 балл')
    
    # === STAGE 2: FWHM width evaluation ===
    print('\nЭТАП 2: Оценка ширины на полувысоте (FWHM)')
    sorted_widths = np.argsort(widths)[::-1]
    
    if n_peaks > 1:
        max_width_idx = int(sorted_widths[0])  # Explicit cast to int
        print(f'  Пик #{max_width_idx + 1}: FWHM = {widths[max_width_idx]:.2f} (максимум) → +0.000 балл')
        
        remaining_indices = sorted_widths[1:].astype(int)
        remaining_widths = widths[remaining_indices]
        width_range = np.max(remaining_widths) - np.min(remaining_widths)
        
        if width_range > 0:
            norm_widths = ((remaining_widths - np.min(remaining_widths)) / width_range) * 0.95
            for i, idx in enumerate(remaining_indices):
                Points[idx] += norm_widths[i]
                print(f'  Пик #{idx + 1}: FWHM = {widths[idx]:.2f}, норм. балл = {norm_widths[i]:.3f}')
        else:
            for idx in remaining_indices:
                Points[idx] += 0.475
                print(f'  Пик #{idx + 1}: FWHM = {widths[idx]:.2f}, норм. балл = 0.475')
    else:
        print(f'  Пик #1: FWHM = {widths[0]:.2f} (единственный) → +0.000 балл')
    
    # === STAGE 3: Height/Width ratio ===
    print('\nЭТАП 3: Соотношение высота/ширина')
    height_values = data[peaks].astype(float)
    height_width_ratio = height_values / widths
    sorted_ratio = np.argsort(height_width_ratio)[::-1]
    
    if n_peaks > 1:
        max_ratio_idx = int(sorted_ratio[0])  # Explicit cast to int
        Points[max_ratio_idx] += 1.0
        print(f'  Пик #{max_ratio_idx + 1}: H/W = {height_width_ratio[max_ratio_idx]:.2f} (максимум) → +1.000 балл')
        
        remaining_indices = sorted_ratio[1:].astype(int)
        remaining_ratios = height_width_ratio[remaining_indices]
        ratio_range = np.max(remaining_ratios) - np.min(remaining_ratios)
        
        if ratio_range > 0:
            norm_ratios = ((remaining_ratios - np.min(remaining_ratios)) / ratio_range) * 0.95
            for i, idx in enumerate(remaining_indices):
                Points[idx] += norm_ratios[i]
                print(f'  Пик #{idx + 1}: H/W = {height_width_ratio[idx]:.2f}, норм. балл = {norm_ratios[i]:.3f}')
        else:
            for idx in remaining_indices:
                Points[idx] += 0.475
                print(f'  Пик #{idx + 1}: H/W = {height_width_ratio[idx]:.2f}, норм. балл = 0.475')
    else:
        Points[0] += 1.0
        print(f'  Пик #1: H/W = {height_width_ratio[0]:.2f} (единственный) → +1.000 балл')
    
    # === STAGE 4: Prominence evaluation ===
    print('\nЭТАП 4: Оценка проминенса')
    sorted_prom = np.argsort(prominences)[::-1]
    
    if n_peaks > 1:
        max_prom_idx = int(sorted_prom[0])  # Explicit cast to int
        Points[max_prom_idx] += 1.0
        print(f'  Пик #{max_prom_idx + 1}: проминенс = {prominences[max_prom_idx]:.2f} (максимум) → +1.000 балл')
        
        remaining_indices = sorted_prom[1:].astype(int)
        remaining_prom = prominences[remaining_indices]
        prom_range = np.max(remaining_prom) - np.min(remaining_prom)
        
        if prom_range > 0:
            norm_prom = ((remaining_prom - np.min(remaining_prom)) / prom_range) * 0.95
            for i, idx in enumerate(remaining_indices):
                Points[idx] += norm_prom[i]
                print(f'  Пик #{idx + 1}: проминенс = {prominences[idx]:.2f}, норм. балл = {norm_prom[i]:.3f}')
        else:
            for idx in remaining_indices:
                Points[idx] += 0.475
                print(f'  Пик #{idx + 1}: проминенс = {prominences[idx]:.2f}, норм. балл = 0.475')
    else:
        Points[0] += 1.0
        print(f'  Пик #1: проминенс = {prominences[0]:.2f} (единственный) → +1.000 балл')
    
    # === FINAL PEAK SELECTION ===
    n_selected = min(n_sizes, n_peaks)
    sorted_indices = np.argsort(Points)[::-1]
    selected_peaks = np.sort(sorted_indices[:n_selected].astype(int))  # Sort selected peaks by position
    
    # === CONSTRUCT FULL RESULTS DATAFRAME ===
    full_table_data = []
    for i in range(n_peaks):
        full_table_data.append({
            'Peaks number': i + 1,
            'Index': int(peaks[i]),
            'Height': float(data[peaks[i]]),
            'Widths FWHM': float(widths[i]),
            'Area': float(areas[i]),
            'Prominice': float(prominences[i]),
            'left_ps': int(left_ips[i]),
            'right_ips': int(right_ips[i]),
            'Mark': float(Points[i]),
            'Selected': '✓' if i in selected_peaks else ''
        })
    
    df_full = pd.DataFrame(full_table_data)
    
    # Display results summary
    print(f'\n{"=" * 120}')
    print('ПОЛНАЯ ТАБЛИЦА ХАРАКТЕРИСТИК ВСЕХ ПИКОВ:')
    print('=' * 120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df_full.to_string(index=False))
    print('=' * 120)
    
    return df_full
