import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Union
import pandas as pd

import matplotlib.pyplot as plt


def parse_frf_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Парсит FRF-файл (XML-формат) с обработкой пропущенных значений
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()

    root = ET.fromstring(xml_content)

    # 1. Обработка матрицы с заменой ошибок
    matrix = []
    for array in root.findall('.//Matrix/ArrayOfDouble'):
        row = []
        for num in array.findall('double'):
            try:
                row.append(float(num.text) if num.text else 0.0)
            except (ValueError, TypeError):
                row.append(0.0)
        matrix.append(row)

    matrix_columns = [f'matrix_col_{i}' for i in range(
        len(matrix[0]))] if matrix else []
    matrix_df = pd.DataFrame(matrix, columns=matrix_columns).fillna(0)

    # 2. Обработка данных каналов с валидацией
    channels_data = []
    for point in root.findall('.//Point'):
        try:
            data = {
                't': float(point.get('t', 0.0)),
                'U': int(point.get('U', 0)),
                'I': float(point.get('I', 0.0)),
            }
            
            ints = []
            for i in point.findall('./Data/int'):
                try:
                    ints.append(int(i.text) if i.text else 0)
                except (ValueError, TypeError):
                    ints.append(0)
            
            # Добавляем 4 канала с заполнением нулями
            for i, ch in enumerate(['dR110', 'dR6G', 'dTAMRA', 'dROX']):
                data[ch] = ints[i] if i < len(ints) else 0

            channels_data.append(data)
            
        except Exception as e:
            print(f"Ошибка обработки Point: {e}")
            continue

    channels_df = pd.DataFrame(channels_data).fillna(0)

    # 3. Обработка метаданных с фильтрацией пустых значений
    metadata = {}
    elements_to_ignore = {'Matrix', 'Data', 'Point'}
    
    for elem in root.iter():
        # === СПЕЦИАЛЬНЫЙ КЕЙС ===
        if elem.tag == 'SpectrCalibration':
            curves = []
            for arr in elem.findall('.//ArrayOfDouble'):
                curve = [
                    float(d.text) for d in arr.findall('double')
                    if d.text
                    ]
                if curve:
                    curves.append(curve)

            if curves:
                metadata['SpectrCalibration'] = curves
            continue
        
        if elem.tag == 'double':
            # --- concentration ---
            conc_attr = elem.get('Concentration')
            try:
                conc_float = float(conc_attr) if conc_attr is not None else None
            except (ValueError, TypeError):
                conc_float = None

            # --- size ---
            size_text = elem.text.strip() if elem.text and elem.text.strip() else None
            try:
                size_int = int(size_text) if size_text is not None else None
            except (ValueError, TypeError):
                size_int = None

            metadata['Concentration'] = conc_float
            metadata['Size'] = size_int

        if elem.tag in elements_to_ignore:
            continue
            
        content = None
        if elem.text and elem.text.strip():
            content = elem.text.strip()
        elif len(elem) > 0:
            content = {
                child.tag: child.text.strip()
                for child in elem 
                if child.text and child.text.strip()
            }
        
        if content:
            metadata[elem.tag] = content

    return matrix_df, channels_df, metadata