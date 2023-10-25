# -*- coding: utf-8 -*-
"""
Various forecasting models.
"""
# Imports
import copy
import logging
import warnings
from collections.abc import Container

import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ForecasterMixin:
    def _validate_clean_index(self, index, warn=False):
        """
        Validate the index of a pandas Series or DataFrame.

        Parameters
        ----------
        index: a pandas DatetimeIndex.

        Returns
        -------
        True if the index is valid, False otherwise.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("The index must be a pandas DatetimeIndex.")

        # Sort index and check again. If it is still not monotonic increasing, raise an error.
        if not index.is_monotonic_increasing:
            index = index.sort_values()
            if index.is_monotonic_increasing:
                if warn:
                    warnings.warn(
                        "The index was not monotonic increasing, but was after sorting."
                    )
            else:
                raise ValueError("The index must be monotonic increasing.")
        inferred_freq = pd.infer_freq(index)
        if index.freq is None:
            if inferred_freq is None:
                raise ValueError("Index freq is not set and could not be inferred.")
            else:
                warnings.warn(
                    (
                        "Index freq is not set, but could be inferred. Setting it to the inferred value of "
                        f"{inferred_freq}."
                    )
                )
                index.freq = inferred_freq
        elif index.freq != inferred_freq:
            warnings.warn(
                "Index freq is set to %s, but was inferred as %s.",
                index.freq,
                inferred_freq,
            )
        return index

    def __repr__(self):
        def filter_containers(d):
            """This iterates through a dict and returns a dict of its elements minus any that are containers.
            Strings are also returned, although they are a type of container."""

            def iscontainer(x):
                if isinstance(x, str):
                    return False
                elif isinstance(x, Container):
                    return True
                else:
                    return False

            return {k: v for k, v in d.items() if not iscontainer(v)}

        attributes_string = [
            f"{k}={v}" for k, v in filter_containers(vars(self)).items()
        ]
        attributes_string = ", ".join(attributes_string)
        return f"{self.__class__.__name__}({attributes_string})"


class PointRegressionForecaster(ForecasterMixin, BaseEstimator):
    """
    Wrapper that facilitates using a regression model with a scikit-learn-like API to forecast a time series. This class
    uses pandas datetime indices to align the target and predictor series, shift the target time series before fitting
    the regression model, and correctly index the predictions with the date / time being forecast. The number of points
    ahead to forecast is specified by the "step" argument during initialization.

    Additionally, the .fit() method of this class stores the last set of values in the training data to enable
    forecasting the value after the end of a training set by calling the .predict() method without arguments.

    Note that all regressors for each forecasted point are in a single row of the predictor DataFrame, so if multiple
    historical points of the target series are to be used as preditors, a series-to-row transformation has to be
    performed before passing the predictors to the .fit() method.

    Forecasting modes
    -----------------

    This class offers multiple strategies for forecasting, which are set by the "forecast_type" initialization argument
    and the use or not of a base forecast, which is activated by the "base_forecast_column" initialization argument.

    forecast_type=="absolute": the regressor forecasts the absolute value of the target series.

    *** The following four models are yet to be implemented!! ***
    No base forecast and forecast_type=="diff_additive": the regressor forecasts the absolute difference between the
    current value of the series and the future value. The value returned by the .predict() method is
    <current value> + <predicted change>.

    No base forecast and forecast_type=="diff_multiplicative": the regressor forecasts the ratio of the future value of
    the series to the current value. The value returned by the .predict() method is <current value> * <predicted ratio>.

    Base forecast and forecast_type=="base_additive": the regressor forecasts the absolute difference between the
    value of the series forecast by the base forecaster and the true future value. The value returned by the .predict()
    method is <base forecast> + <predicted correction>.

    Base forecast and forecast_type=="base_multiplicative": the regressor forecasts the ratio of the value of the
    series forecast by the base forecaster and the true future value. The value returned by the .predict() method is
    <base forecast> * <predicted correction>.
    """

    def __init__(
        self,
        regressor,
        step: int = 1,
        forecast_type: str = "absolute",
        base_forecast_column=None,
    ):
        """
        Initialize the forecaster.

        Parameters
        ----------
        regressor: a regressor with a scikit-learn-like API, i.e. a .fit() method and a .predict() method.
        step: the number of points ahead to forecast.
        forecast_type: the type of forecast to perform. See the class documentation for details.
        base_forecast_column: if not None, the prefix of the column in the predictors DataFrame that contains the
            base forecast. See the class documentation for details.
        """
        self.regressor = copy.deepcopy(
            regressor
        )  # Regressor is copied to avoid being used by multiple forecasters
        self.step = step
        self.forecast_type = forecast_type
        self.base_forecast_column = base_forecast_column
        self.last_predictors = None

    def fit(self, predictors, target):
        """
        Fit the forecaster.

        Parameters
        ----------
        predictors: a pandas DataFrame with the predictors. The index must be a pandas DatetimeIndex.
        target: Either a pandas Series with the series to be forecast, or a string with the name of a column in the
        predictors DataFrame.
        """
        # If target is a string, extract this column from the predictors.
        if isinstance(target, str):
            target = predictors[target]
        # Validate the index of the predictors
        predictors.index = self._validate_clean_index(predictors.index)
        # Validate the index of the target
        predictors.index = self._validate_clean_index(target.index)
        # Shift the target
        target = target.shift(-self.step).dropna()
        # Store the last values of the predictors
        self.last_predictors = predictors.loc[[predictors.index.max()], :]
        self.last_predictors.index.freq = predictors.index.freq
        if self.forecast_type == "absolute":
            # Align the predictors and target
            predictors, target = predictors.align(target, join="inner", axis=0)
            # Fit the regressor
            self.regressor = self.regressor.fit(predictors, target)
        else:
            raise NotImplementedError("Forecast type not implemented yet.")

    def predict(self, predictors=None):
        """
        Predict the target series.

        Parameters
        ----------
        predictors: a pandas DataFrame with the predictors. The index must be a pandas DatetimeIndex. If None, the
            last set of predictors used in the .fit() method is used.

        Returns
        -------
        A pandas Series with the predicted values.
        """
        if predictors is None:
            predictors = self.last_predictors
        if self.forecast_type == "absolute":
            # Predict the target
            predictions = self.regressor.predict(predictors)
        else:
            raise NotImplementedError("Forecast type not implemented yet.")
        # Calculate the index of the predictions
        index = predictors.index + self.step * predictors.index.freq
        columns = [f"forecast_lead{self.step}"]
        # Create a pandas DataFrame with the predictions
        predictions = pd.DataFrame(data=predictions, index=index, columns=columns)
        return predictions

    def fit_predict(self, predictors, target):
        """
        Fit the forecaster and predict the target series.
        """
        self.fit(predictors, target)
        return self.predict()


class MultiPointRegressionForecaster(ForecasterMixin):
    """
    Contains multiple instances of PointRegressionForecaster which forecast up to a specific forecast horizon
    using a specific stride.
    """

    def __init__(
        self,
        regressor,
        horizon: int,
        stride: int = 1,
        forecast_type: str = "absolute",
        base_forecast_column=None,
    ):
        self.regressor = regressor
        self.horizon = horizon
        self.stride = stride
        self.forecast_type = forecast_type
        self.base_forecast_column = base_forecast_column

        # Check that stride is a factor of horizon, raise an error if not.
        if self.horizon % self.stride != 0:
            raise ValueError("Stride must be a factor of horizon.")

        # Create a list of instances of PointRegressionForecaster with their step parameter corresponding to the
        # horizon and stride.
        self.forecasters = []
        for step in range(stride, horizon + 1, stride):
            self.forecasters.append(
                PointRegressionForecaster(
                    regressor,
                    step=step,
                    forecast_type=forecast_type,
                    base_forecast_column=base_forecast_column,
                )
            )

    def fit(self, X, y):
        """
        Fits the forecaster to the training data.

        Parameters:
            X (array-like): The input data.
            y (array-like): The target values.

        Returns:
            None
        """
        for forecaster in self.forecasters:
            forecaster.fit(X, y)

    def predict(self, X=None):
        """
        Predicts the values using the specified input data.

        Parameters:
            X (DataFrame, optional): The input data for prediction. If not provided, the function uses the internal
            data stored in the object, which corresponds to the last timepoint in the training data.

        Raises:
            ValueError: If multiple rows of X are provided (method currently only supports forecasts from a single
            start date).

        Returns:
            forecast (Series): The predicted values.
        """
        if X is not None and X.shape[0] > 1:
            raise ValueError(
                "Forecasting from more than one start dates not currently supported. X must be a single row."
            )
        predictions = {}
        for forecaster in self.forecasters:
            prediction = forecaster.predict(X)
            predictions[prediction.index[0]] = prediction.values[0][0]
        forecast = pd.Series(predictions, name="forecast")

        return forecast
