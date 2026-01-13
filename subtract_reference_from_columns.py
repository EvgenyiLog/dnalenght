from typing import Literal, Optional
import pandas as pd
import numpy as np


def subtract_reference_from_columns(
    df: pd.DataFrame,
    n: int,
    method: Literal["mean", "percentile"] = "mean",
    percentile: Optional[float] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Subtract a reference value (mean or percentile) computed from the first N elements
    of each column from the entire column.

    The operation is performed independently for each column.
    Column names and index are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric columns.
    n : int
        Number of initial rows used to compute the reference value.
        Must be >= 1.
    method : {"mean", "percentile"}, default="mean"
        Method used to compute the reference value:
        - "mean": arithmetic mean of the first N values
        - "percentile": percentile of the first N values
    percentile : float, optional
        Percentile value in the range [0, 100].
        Required if method="percentile".
    inplace : bool, default=False
        If True, modifies the input DataFrame in place.
        If False, returns a new DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with reference value subtracted from each column.

    Raises
    ------
    ValueError
        If n < 1 or percentile is not provided when required.
    TypeError
        If non-numeric columns are encountered.

    Notes
    -----
    - NaN values in the first N elements are ignored when computing
      mean or percentile.
    - Common use cases:
        * baseline correction
        * background subtraction
        * signal normalization
    """

    if n < 1:
        raise ValueError("Parameter 'n' must be >= 1.")

    if method == "percentile" and percentile is None:
        raise ValueError("Parameter 'percentile' must be specified when method='percentile'.")

    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise TypeError("All columns must be numeric.")

    target = df if inplace else df.copy()

    for col in target.columns:
        head_values = target[col].iloc[:n].to_numpy()

        if method == "mean":
            ref_value = np.nanmean(head_values)
        else:
            ref_value = np.nanpercentile(head_values, percentile)

        target[col] = target[col] - ref_value

    return target
