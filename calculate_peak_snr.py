import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from scipy import signal as scipy_signal


def calculate_peak_snr(
    signal: Union[np.ndarray, pd.Series],
    peak_indices: Union[List[int], np.ndarray],
    noise_window: int = 50,
    method: str = "local",
    baseline: Optional[Union[np.ndarray, pd.Series]] = None,
    min_noise_std: float = 1e-6
) -> Dict[str, Any]:
    """
    Рассчитывает отношение сигнал/шум (SNR) для каждого пика в сигнале.
    
    Пиковый SNR определяется как отношение амплитуды пика над базовой линией
    к стандартному отклонению шума в локальном окне.
    
    Формула:
        SNR = (peak_amplitude - baseline_value) / noise_std
    
    Параметры
    ----------
    signal : np.ndarray или pd.Series
        Исходный сигнал (временной ряд значений интенсивности).
    peak_indices : list[int] или np.ndarray
        Индексы позиций пиков в сигнале (отсортированы по возрастанию).
    noise_window : int, optional (default=50)
        Размер окна (в точках) для расчёта шума слева и справа от пика.
        Используется только при method="local".
    method : str, optional (default="local")
        Метод расчёта шума:
        - "local": шум рассчитывается в окнах слева/справа от каждого пика
        - "global": шум рассчитывается по всему сигналу (исключая окна пиков ±2*noise_window)
        - "baseline": шум рассчитывается как отклонение от переданной базовой линии
    baseline : np.ndarray или pd.Series, optional (default=None)
        Предварительно вычисленная базовая линия сигнала.
        Если не задана — вычисляется автоматически методом морфологической фильтрации.
    min_noise_std : float, optional (default=1e-6)
        Минимальное значение стандартного отклонения шума для избежания деления на ноль.
    
    Возвращает
    ----------
    dict : Словарь с результатами:
        - "snr_values": np.ndarray — SNR для каждого пика
        - "peak_amplitudes": np.ndarray — амплитуды пиков над базовой линией
        - "noise_std_values": np.ndarray — стандартное отклонение шума для каждого пика
        - "baseline_values": np.ndarray — значения базовой линии под пиками
        - "noise_windows": list[tuple] — координаты окон расчёта шума [(left_start, left_end), ...]
    
    Примеры
    --------
    >>> signal = np.random.randn(1000) + np.sin(np.linspace(0, 10, 1000)) * 10
    >>> peaks, _ = scipy_signal.find_peaks(signal, height=5)
    >>> result = calculate_peak_snr(signal, peaks)
    >>> print(f"SNR первого пика: {result['snr_values'][0]:.2f}")
    
    >>> # Для колонки DataFrame
    >>> df = pd.DataFrame({"intensity": signal})
    >>> result = calculate_peak_snr(df["intensity"], peaks, method="global")
    
    Примечания
    ----------
    1. Для электрофореза ДНК рекомендуется:
       - метод="local" для изолированных пиков
       - метод="global" для плотных фрагментов с перекрытием
    
    2. Типичные пороги качества:
       - SNR < 3: пик не надёжен (ниже предела обнаружения)
       - 3 ≤ SNR < 10: пик обнаружен, но количественно неточен
       - SNR ≥ 10: надёжный пик для количественного анализа
    
    3. Автоматическая базовая линия вычисляется методом:
       `scipy.ndimage.minimum_filter1d` с окном 50 точек + медианная фильтрация.
    """
    # === Валидация входных данных ===
    if len(peak_indices) == 0:
        raise ValueError("Список пиков пуст")
    
    # Конвертируем в numpy массив для единообразной обработки
    signal_arr = np.asarray(signal, dtype=np.float64)
    peak_indices_arr = np.asarray(peak_indices, dtype=np.int64)
    
    if signal_arr.ndim != 1:
        raise ValueError(f"Сигнал должен быть одномерным, получено: {signal_arr.ndim}D")
    
    if len(signal_arr) < 10:
        raise ValueError("Сигнал слишком короткий (< 10 точек)")
    
    # Сортируем пики и удаляем дубликаты
    peak_indices_arr = np.unique(np.sort(peak_indices_arr))
    
    # Проверяем границы пиков
    if np.any(peak_indices_arr < 0) or np.any(peak_indices_arr >= len(signal_arr)):
        raise ValueError("Некоторые пики выходят за границы сигнала")
    
    # === Вычисление базовой линии (если не передана) ===
    if baseline is None:
        baseline_arr = _estimate_baseline(signal_arr)
    else:
        baseline_arr = np.asarray(baseline, dtype=np.float64)
        if baseline_arr.shape != signal_arr.shape:
            raise ValueError(
                f"Форма базовой линии {baseline_arr.shape} не совпадает с сигналом {signal_arr.shape}"
            )
    
    # === Расчёт амплитуд пиков над базовой линией ===
    peak_amplitudes = signal_arr[peak_indices_arr] - baseline_arr[peak_indices_arr]
    
    # === Расчёт шума в зависимости от метода ===
    if method == "local":
        noise_std_values, noise_windows = _calculate_local_noise(
            signal_arr, baseline_arr, peak_indices_arr, noise_window
        )
    elif method == "global":
        global_std = _calculate_global_noise(signal_arr, baseline_arr, peak_indices_arr, noise_window)
        noise_std_values = np.full(len(peak_indices_arr), global_std)
        noise_windows = [(0, len(signal_arr))] * len(peak_indices_arr)
    elif method == "baseline":
        noise_std_values = np.abs(signal_arr - baseline_arr).std()
        noise_std_values = np.full(len(peak_indices_arr), noise_std_values)
        noise_windows = [(0, len(signal_arr))] * len(peak_indices_arr)
    else:
        raise ValueError(
            f"Неизвестный метод расчёта шума: '{method}'. "
            f"Допустимые значения: 'local', 'global', 'baseline'"
        )
    
    # === Расчёт пикового SNR ===
    # Защита от деления на ноль
    noise_std_values = np.maximum(noise_std_values, min_noise_std)
    snr_values = peak_amplitudes / noise_std_values
    
    return {
        "snr_values": snr_values,
        "peak_amplitudes": peak_amplitudes,
        "noise_std_values": noise_std_values,
        "baseline_values": baseline_arr[peak_indices_arr],
        "noise_windows": noise_windows
    }


def _estimate_baseline(signal: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Оценивает базовую линию сигнала методом морфологической фильтрации.
    
    Использует комбинацию минимальной фильтрации и медианной сглаживания
    для устойчивой оценки базовой линии в присутствии пиков.
    
    Параметры
    ----------
    signal : np.ndarray
        Исходный сигнал.
    window_size : int
        Размер окна фильтрации (в точках).
    
    Возвращает
    ----------
    np.ndarray
        Оценка базовой линии той же длины, что и сигнал.
    """
    from scipy.ndimage import minimum_filter1d, median_filter
    
    # Шаг 1: Минимальная фильтрация для подавления пиков
    min_filtered = minimum_filter1d(signal, size=window_size)
    
    # Шаг 2: Медианная фильтрация для сглаживания
    baseline = median_filter(min_filtered, size=window_size // 2)
    
    # Шаг 3: Гарантируем, что базовая линия не выше сигнала
    baseline = np.minimum(baseline, signal)
    
    return baseline


def _calculate_local_noise(
    signal: np.ndarray,
    baseline: np.ndarray,
    peak_indices: np.ndarray,
    window_size: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Рассчитывает локальный шум в окнах слева и справа от каждого пика.
    
    Для каждого пика:
    1. Берётся окно слева: [peak - 2*window_size, peak - window_size]
    2. Берётся окно справа: [peak + window_size, peak + 2*window_size]
    3. Шум = std(сигнал - базовая_линия) в объединённом окне
    
    Возвращает
    ----------
    tuple : (noise_std_values, noise_windows)
        - noise_std_values: np.ndarray — std шума для каждого пика
        - noise_windows: list[tuple] — координаты окон [(left_start, left_end), ...]
    """
    noise_std_values = np.zeros(len(peak_indices))
    noise_windows = []
    
    for i, peak_idx in enumerate(peak_indices):
        # Определяем границы окон
        left_start = max(0, peak_idx - 2 * window_size)
        left_end = max(0, peak_idx - window_size)
        right_start = min(len(signal), peak_idx + window_size)
        right_end = min(len(signal), peak_idx + 2 * window_size)
        
        # Собираем точки шума из левого и правого окон
        noise_points = []
        if left_end > left_start:
            noise_points.extend(signal[left_start:left_end] - baseline[left_start:left_end])
        if right_end > right_start:
            noise_points.extend(signal[right_start:right_end] - baseline[right_start:right_end])
        
        # Если нет точек шума — используем глобальный шум
        if len(noise_points) < 5:
            global_noise = np.std(signal - baseline)
            noise_std = global_noise if global_noise > 0 else 1.0
        else:
            noise_std = np.std(noise_points)
        
        noise_std_values[i] = noise_std
        noise_windows.append((left_start, left_end, right_start, right_end))
    
    return noise_std_values, noise_windows


def _calculate_global_noise(
    signal: np.ndarray,
    baseline: np.ndarray,
    peak_indices: np.ndarray,
    exclusion_window: int
) -> float:
    """
    Рассчитывает глобальный шум по всему сигналу, исключая окна вокруг пиков.
    
    Параметры
    ----------
    signal : np.ndarray
        Исходный сигнал.
    baseline : np.ndarray
        Базовая линия.
    peak_indices : np.ndarray
        Индексы пиков.
    exclusion_window : int
        Размер окна (в точках) для исключения вокруг каждого пика.
    
    Возвращает
    ----------
    float
        Стандартное отклонение шума в "чистых" участках сигнала.
    """
    # Создаём маску для исключения окон вокруг пиков
    mask = np.ones(len(signal), dtype=bool)
    for peak_idx in peak_indices:
        start = max(0, peak_idx - exclusion_window)
        end = min(len(signal), peak_idx + exclusion_window)
        mask[start:end] = False
    
    # Рассчитываем шум только в "чистых" участках
    noise_points = signal[mask] - baseline[mask]
    
    if len(noise_points) < 10:
        # Если мало точек — используем весь сигнал
        noise_points = signal - baseline
    
    return np.std(noise_points)


# === Пример использования ===
if __name__ == "__main__":
    # Генерируем тестовый сигнал с пиками
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    signal = np.sin(x * 5) * 10 + np.random.randn(1000) * 0.5
    
    # Добавляем "пики" ДНК-фрагментов
    peaks_positions = [150, 300, 450, 600, 750]
    for pos in peaks_positions:
        signal[pos-10:pos+10] += np.exp(-((np.arange(20) - 10) ** 2) / 20) * 50
    
    # Находим пики
    peaks, _ = scipy_signal.find_peaks(signal, height=15, distance=50)
    
    # Рассчитываем SNR
    result = calculate_peak_snr(
        signal,
        peaks,
        noise_window=30,
        method="local"
    )
    
    # Выводим результаты
    print("Результаты расчёта пикового SNR:")
    print("=" * 60)
    for i, (idx, snr, amp, noise) in enumerate(zip(
        peaks, 
        result["snr_values"], 
        result["peak_amplitudes"], 
        result["noise_std_values"]
    )):
        quality = "✅" if snr >= 10 else "⚠️" if snr >= 3 else "❌"
        print(f"{quality} Пик #{i+1:2d} (позиция {idx:4d}): "
              f"амплитуда={amp:6.2f}, шум={noise:5.3f}, SNR={snr:6.2f}")
    
    print("=" * 60)
    print(f"Средний SNR: {np.mean(result['snr_values']):.2f}")
    print(f"Минимальный SNR: {np.min(result['snr_values']):.2f}")