# Functions and methods for manipulating data and engineering features use in time series forecasting.
from scipy.interpolate import UnivariateSpline

import pandas as pd


def create_lagged_values(df, n, column: str = None):
    """
    Create lagged values of a specified column in a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        n (int): The number of lagged values to create.
        column (str, optional): The name of the column to be lagged. Obligatory if more than one column. Defaults to
        None.

    Returns:
        DataFrame: The DataFrame with the lagged values added.

    Raises:
        ValueError: If column is not provided and the input DataFrame has more than one column.

    """
    if column is None:
        if df.shape[1] > 1:
            raise ValueError(
                "Column name must be provided if more than one column is provided."
            )
        column = df.columns[0]
    for i in range(1, n + 1):
        df[f"{column}_lag{i}"] = df[column].shift(i)
    return df


def smooth_forecast_univariate(
    forecast: pd.Series,
    historical: pd.Series = None,
    include_last_n: int = 1,
    freq: str = None,
    *,
    k=2,
    **kwargs,
) -> pd.Series:
    """
    Smoothens a univariate time series (forecast or any series) using UnivariateSpline.

    Parameters:
        forecast (Series): The time series to be smoothened.
        historical (Series, optional): A historical series that can be used in combination with forecast.
        include_last_n (int, optional): Number of historical points to include with the forecast for smoothing.
                                        Defaults to 1.
        freq (str, optional): Frequency of the time series. If not provided, it will be inferred.
        k (int, optional): Degree of the spline. Defaults to 2.

    Returns:
        Series: The smoothened series.

    Raises:
        ValueError: If frequency is not specified and cannot be inferred from the series.
    """

    # Check and set frequency
    if freq is None:
        freq = (
            historical.index.freq or pd.infer_freq(historical.index)
            if historical is not None
            else None
        )
    if freq is None:
        freq = forecast.index.freq or pd.infer_freq(forecast.index)
    if freq is None:
        raise ValueError(
            "Output frequency must either be specified or inferable from the provided input series."
        )

    # Prepare data for the spline fitting
    if historical is not None and include_last_n > 0:
        smoother_fit_input = pd.concat((historical[-include_last_n:], forecast))
    else:
        smoother_fit_input = forecast

    # Apply UnivariateSpline for smoothing
    spl = UnivariateSpline(
        smoother_fit_input.index.astype(int), smoother_fit_input, k=k, **kwargs
    )

    output_index = pd.date_range(
        start=smoother_fit_input.index[0], end=smoother_fit_input.index[-1], freq=freq
    )
    output = pd.Series(
        spl(output_index.astype(int)), index=output_index, name="smoothed_forecast"
    )

    return output
