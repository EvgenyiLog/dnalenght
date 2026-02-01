from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from typing import Any

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]

    elif isinstance(obj, tuple):
        return [convert_numpy_types(v) for v in obj]

    elif isinstance(obj, set):
        return [convert_numpy_types(v) for v in obj]

    elif isinstance(obj, (Path, PurePath)):
        return str(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif isinstance(obj, (np.bool_,)):
        return bool(obj)

    elif isinstance(obj, np.generic):
        return obj.item()

    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    elif isinstance(obj, pd.Series):
        return obj.tolist()

    return obj