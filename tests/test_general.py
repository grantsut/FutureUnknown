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
from futureunknown.data import create_lagged_values


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
        index = ForecasterMixin()._validate_clean_index(index)
        assert index.is_monotonic_increasing


# Test feature engineering code
def test_create_lagged_values():
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


# Test the PointRegressionForecaster class.
def test_point_regression_forecaster():
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


class test_forecaster_accuracy:
    def test_linear_trend():
        # Baseic test of accurate output with a LinearRegression forecaster
        from futureunknown import regression
        from sklearn.linear_model import LinearRegression

        model = regression.MultiPointRegressionForecaster(
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
