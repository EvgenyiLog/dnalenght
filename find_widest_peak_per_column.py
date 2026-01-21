import pandas as pd
import numpy as np
from scipy.signal import argrelmax
from typing import Dict, List, Tuple
import warnings

def find_widest_peak_per_column(df: pd.DataFrame, order: int = 1, height_factor: float = 0.5) -> Dict[str, Dict]:
    """
    Находит В КАЖДОЙ КОЛОНКЕ самый широкий пик и его подпики.
    
    Для каждой числовой колонки:
    1. Находит все пики с помощью argrelmax
    2. Определяет ширину каждого пика (от половины амплитуды)
    3. Выбирает САМЫЙ ШИРОКИЙ пик в колонке
    4. В этом пике ищет подпики от половины максимальной амплитуды
    
    Args:
        df (pd.DataFrame): Входной DataFrame с числовыми данными
        order (int): Параметр order для argrelmax
        height_factor (float): Фактор высоты для определения границ пика (0.5 = половина)
    
    Returns:
        Dict[str, Dict]: {колонка: {
            'widest_peak': (индекс, значение),
            'peak_width': ширина_пика,
            'subpeaks': [(индекс, значение), ...]
        }}
    """
    results = {}
    
    # Обрабатываем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        series = df[col].dropna().values
        if len(series) < order * 2 + 1:
            warnings.warn(f"Колонка {col}: недостаточно данных")
            continue
        
        # 1. Находим все пики в колонке
        peak_indices = argrelmax(series, order=order)[0]
        peaks = [(i, series[i]) for i in peak_indices]
        
        if not peaks:
            warnings.warn(f"Колонка {col}: пики не найдены")
            continue
        
        # 2. Для каждого пика определяем ширину
        peak_widths = []
        for peak_idx, peak_val in peaks:
            half_height = peak_val * height_factor
            
            # Левая граница пика
            left_bound = peak_idx
            while (left_bound > order and 
                   series[left_bound] >= half_height):
                left_bound -= 1
            
            # Правая граница пика
            right_bound = peak_idx
            while (right_bound < len(series) - 1 - order and 
                   series[right_bound] >= half_height):
                right_bound += 1
            
            width = right_bound - left_bound + 1
            peak_widths.append((peak_idx, peak_val, width, left_bound, right_bound))
        
        # 3. Находим САМЫЙ ШИРОКИЙ пик в этой колонке
        widest_peak = max(peak_widths, key=lambda x: x[2])
        widest_idx, widest_val, widest_width, left_b, right_b = widest_peak
        
        # 4. Ищем ПОДПИКИ в области самого широкого пика
        peak_region = series[left_b:right_b + 1]
        if len(peak_region) > order * 2 + 1:
            subpeak_rel_indices = argrelmax(peak_region, order=order)[0]
            subpeaks = [(left_b + i, peak_region[i]) for i in subpeak_rel_indices]
        else:
            subpeaks = []
        
        # Сохраняем результат для колонки
        results[col] = {
            'widest_peak': (widest_idx, widest_val),
            'peak_width': widest_width,
            'peak_bounds': (left_b, right_b),
            'subpeaks': subpeaks
        }
    
    return results

# Пример использования:
if __name__ == "__main__":
    # Генерируем данные с разными пиковыми структурами
    t = np.linspace(0, 10, 2000)
    
    # Первая колонка - широкий пик с подпиками
    signal1 = 2 * np.exp(-((t-3)/1.5)**2) + 1.2 * np.exp(-((t-3.2)/0.4)**2) + 0.8 * np.exp(-((t-2.8)/0.3)**2)
    
    # Вторая колонка - более узкие пики
    signal2 = np.sin(2 * np.pi * t * 0.8) + 0.5 * np.sin(6 * np.pi * t)
    
    df = pd.DataFrame({'signal_wide': signal1, 'signal_narrow': signal2})
    
    results = find_widest_peak_per_column(df, order=5, height_factor=0.5)
    
    for col, info in results.items():
        print(f"\nКолонка '{col}':")
        print(f"  Самый широкий пик: {info['widest_peak']} (ширина: {info['peak_width']})")
        print(f"  Границы пика: {info['peak_bounds']}")
        print(f"  Подпики: {len(info['subpeaks'])} шт. {info['subpeaks']}")
