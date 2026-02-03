from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from typing import Any


def find_numpy_types(obj: Any, path="root"):
    """Рекурсивно ищет numpy типы в структуре данных"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_numpy_types(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_numpy_types(v, f"{path}[{i}]")
    elif isinstance(obj, (np.integer, np.floating, np.bool_, np.ndarray)):
        print(f"⚠️  Numpy type found at {path}: {type(obj).__name__} = {obj}")
