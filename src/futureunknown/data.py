# Functions and methods for manipulating data and engineering features.
import pandas as pd

def create_lagged_values(df, n, column: str=None):
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
            raise ValueError("Column name must be provided if more than one column is provided.")
        column = df.columns[0]
    for i in range(1, n+1):
        df[f"{column}_lag{i}"] = df[column].shift(i)
    return df
