# Initial file for tests. Will be split as the number of tests grows.

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from futureunknown.regression import (
    ForecasterMixin,
    PointRegressionForecaster,
    MultiPointRegressionForecaster,
)

from futureunknown.data import create_lagged_values, smooth_forecast_univariate


class Test_index_validator:
    """Test the _validate_clean_index method of the ForecasterMixin class with various types of good and bad indices."""

    def test_good_daily_index(self):
        """Test a good daily index."""
        index = pd.date_range(start="2019-01-01", end="2019-01-31", freq="D")
        assert (ForecasterMixin()._validate_clean_index(index) == index).all()

    def test_good_monthly_index(self):
        """Test a good monthly index."""
        index = pd.date_range(start="2019-01-01", end="2019-12-31", freq="M")
        assert (ForecasterMixin()._validate_clean_index(index) == index).all()

    def test_good_business_monthly_index(self):
        """Test a good business monthly index."""
        index = pd.date_range(start="2019-01-01", end="2019-12-31", freq="BM")
        assert (ForecasterMixin()._validate_clean_index(index) == index).all()

    def test_bad_daily_index(self):
        """Test a daily index in which dates are missing."""
        index = pd.date_range(start="2019-01-01", end="2019-01-31", freq="D")
        index = index.drop(index[3])
        with pytest.raises(ValueError):
            ForecasterMixin()._validate_clean_index(index)

    def test_non_datetime_index(self):
        """Test an index that is not a DatetimeIndex."""
        index = pd.Index([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            ForecasterMixin()._validate_clean_index(index)

    def test_non_monotonic_index(self):
        """Test that an incorrectly-sorted index is fixed."""
        index = pd.date_range(start="2019-01-01", end="2019-01-31", freq="D")
        index = index[::-1]
        index = ForecasterMixin()._validate_clean_index(index, warn=False)
        assert index.is_monotonic_increasing


class Test_forecaster_accuracy:
    """Tests for evaluating the accuracy of the forecaster."""

    def test_linear_trend(self):
        """Basic test of accurate output with a LinearRegression forecaster."""
        model = MultiPointRegressionForecaster(
            LinearRegression(), horizon=3, stride=1, forecast_type="absolute"
        )

        X = pd.DataFrame(
            data=list(range(10)),
            index=pd.date_range("2023-01-01", periods=10),
            columns=["a"],
        )
        X = create_lagged_values(X, 4, "a").dropna()
        y = X.iloc[:, 0]

        model.fit(X, y)
        predictions = model.predict()
        assert np.allclose(predictions, [10, 11, 12])


class Test_smooth_forecast_univariate:
    """Tests for smooth_forecast_univariate()."""

    def test_basic_forecast_only(self):
        """Test that output of the correct length is created."""
        forecast = pd.Series(
            [1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5)
        )
        smoothed = smooth_forecast_univariate(forecast, freq="D")
        assert len(smoothed) == len(forecast)

    def test_with_historical(self):
        """Test that output of the correct length is created when smoothing the forecast and the end of a historical
        series."""
        historical = pd.Series(
            [0.5, 0.75, 1], index=pd.date_range(start="2023-01-01", periods=3, freq="D")
        )
        forecast = pd.Series(
            [2, 3, 4, 5], index=pd.date_range(start="2023-01-04", periods=4, freq="D")
        )
        smoothed = smooth_forecast_univariate(
            forecast, historical=historical, include_last_n=2
        )
        assert len(smoothed) == len(forecast) + 2

    def test_without_specifying_freq(self):
        """Test that an error is raised if the frequency is not specified and can not be inferred. """
        forecast = pd.Series(
            [1, 2, 3, 4, 5, 6], index=pd.date_range("2023-01-01", periods=6)
        )
        forecast = forecast.iloc[[0, 2, 5]]
        with pytest.raises(ValueError):
            smooth_forecast_univariate(forecast)

    def test_output_frequency(self):
        """ Test that interpolation works correctly. """
        forecast = pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range(start="2023-01-01", periods=5, freq="2D"),
        )
        smoothed = smooth_forecast_univariate(forecast, freq="D")
        inferred_freq = pd.infer_freq(smoothed.index)
        assert inferred_freq == "D"


# Test the PointRegressionForecaster class.
def test_point_regression_forecaster():
    """Test the functionality of the PointRegressionForecaster."""
    # Initialize a Linear Regression model
    model = LinearRegression()

    # Initialize the PointRegressionForecaster
    forecaster = PointRegressionForecaster(model, step=1)

    # Create some dummy data
    predictors = pd.DataFrame(
        {"a": range(10)}, index=pd.date_range("2023-01-01", periods=10)
    )
    target = pd.Series(range(10), index=pd.date_range("2023-01-01", periods=10))

    # Fit the forecaster
    forecaster.fit(predictors, target)

    # Test prediction
    prediction = forecaster.predict(predictors.iloc[[0], :])
    assert prediction.index[0] == pd.Timestamp("2023-01-02")
    assert prediction.columns[0] == "forecast_lead1"


def test_multi_point_regression_forecaster():
    """Test the functionality of the MultiPointRegressionForecaster."""
    # Initialize a Linear Regression model
    model = LinearRegression()

    # Initialize the MultiPointRegressionForecaster
    forecaster = MultiPointRegressionForecaster(model, horizon=3, stride=1)

    # Create some dummy data
    predictors = pd.DataFrame(
        {"a": range(10)}, index=pd.date_range("2023-01-01", periods=10)
    )
    target = pd.Series(range(10), index=pd.date_range("2023-01-01", periods=10))

    # Fit the forecaster
    forecaster.fit(predictors, target)

    # Test prediction
    prediction = forecaster.predict(predictors.iloc[[0], :])
    assert prediction.index[0] == pd.Timestamp("2023-01-02")
    assert prediction.index[1] == pd.Timestamp("2023-01-03")
    assert prediction.index[2] == pd.Timestamp("2023-01-04")

    # Test ValueError when stride is not a factor of horizon
    with pytest.raises(ValueError):
        forecaster = MultiPointRegressionForecaster(model, horizon=5, stride=2)


def test_create_lagged_values():
    """Test the creation of lagged values for a given dataset."""
    # Case 1: DataFrame has only one column and no column name is provided
    df_single_col = pd.DataFrame({"a": range(10)})
    result = create_lagged_values(df_single_col, 2)
    assert list(result.columns) == ["a", "a_lag1", "a_lag2"]

    # Case 2: DataFrame has more than one column and a column name is provided
    df_multi_col = pd.DataFrame({"a": range(10), "b": range(10, 20)})
    result = create_lagged_values(df_multi_col, 2, "a")
    assert list(result.columns) == ["a", "b", "a_lag1", "a_lag2"]

    # Case 3: DataFrame has more than one column and no column name is provided
    with pytest.raises(ValueError):
        create_lagged_values(df_multi_col, 2)

    # Case 4: DataFrame has only one column and the column name provided doesn't exist in the DataFrame
    with pytest.raises(KeyError):
        create_lagged_values(df_single_col, 2, "b")
