# Functions and methods for manipulating data and engineering features.
from scipy.interpolate import UnivariateSpline

import pandas as pd


def create_lagged_values(df, n, column: str = None):
    """
    Create lagged values of a specified column in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        n (int): The number of lagged values to create.
        column (str, optional): The name of the column to be lagged. Obligatory if more than one column. Defaults to None.

    Returns:
        DataFrame: The DataFrame with the lagged values.

    Raises:
        ValueError: If column is not provided and the input DataFrame has more than one column.

    """
    if column is None:
        if df.shape[1] > 1:
            raise ValueError(
                "Column name must be provided if more than one column is provided.")
        column = df.columns[0]
    for i in range(1, n+1):
        df[f"{column}_lag{i}"] = df[column].shift(i)
    return df


def smooth_forecast_univariate(forecast: pd.Series, historical: pd.Series = None, include_last_n: int = 1, freq: str = None, *, k=2, **kwargs):
    if freq is None:
        if forecast.index.freq is None:
            freq = pd.infer_freq(historical.index)
        else:
            freq = historical.index.freq
        if freq is None:
            raise ValueError(
                "Output frequency must either be specified or inferable from the provided input series.")

    if historical is not None and include_last_n > 0:
        smoother_fit_input = pd.concat(
            (historical[-include_last_n:], forecast))
    else:
        smoother_fit_input = forecast

    spl = UnivariateSpline(smoother_fit_input.index.astype(
        int), smoother_fit_input, k=k, **kwargs)

    output_index = pd.date_range(
        start=smoother_fit_input.index[0], end=smoother_fit_input.index[-1], freq=freq)
    output = pd.Series(spl(output_index.astype(int)), index=output_index, name="smoothed_forecast")
    return output
