# -*- coding: utf-8 -*-
"""
Various forecasting models.
"""
# Imports
import logging
import warnings

import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ForecasterMixin():

    def _validate_clean_index(self, index):
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
                logger.warning(
                    "The index was not monotonic increasing, but was after sorting.")
            else:
                raise ValueError("The index must be monotonic increasing.")
        inferred_freq = pd.infer_freq(index)
        if index.freq is None:
            if inferred_freq is None:
                raise ValueError(
                    "Index freq is not set and could not be inferred.")
            else:
                logger.warning(
                    "Index freq is not set, but could be inferred. Setting it to the inferred value of %s.",
                    inferred_freq
                )
                index.freq = inferred_freq
        elif index.freq != inferred_freq:
            logger.warning(
                "Index freq is set to %s, but was inferred as %s.", index.freq, inferred_freq)
        return index


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
    and the use or not of a base forecast, which is activated by the "base_forecast_prefix" initialization argument.

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

    def __init__(self, regressor, step: int = 1, forecast_type: str = "absolute", base_forecast_column=None):
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
        self.regressor = regressor
        self.step = step
        self.forecast_type = forecast_type
        self.base_forecast_prefix = base_forecast_column
        self.last_predictors = None

    def fit(self, predictors, target):
        """
        Fit the forecaster.

        Parameters
        ----------
        predictors: a pandas DataFrame with the predictors. The index must be a pandas DatetimeIndex.
        target: a pandas Series with the target. The index must be a pandas DatetimeIndex.
        """
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
        predictions = pd.DataFrame(
            data=predictions, index=index, columns=columns)
        return predictions

    def fit_predict(self, predictors, target):
        """
        Fit the forecaster and predict the target series.
        """
        self.fit(predictors, target)
        return self.predict()

    class MultiPointRegressionForecaster(ForecasterMixin, BaseEstimator):
        """
        Contains multiple instances of PointRegressionForecaster which forecast up to a specific forecast horizon
        using a specific stride.
        """

        def __init__(self, regressor, horizon, stride: int = 1, forecast_type: str = "absolute", base_forecast_column=None):
            self.regressor = regressor
            self.horizon = horizon
            self.stride = stride
            self.forecast_type = forecast_type
            self.base_forecast_prefix = base_forecast_column

            # Check that stride is a factor of horizon, raise an error if not.
            if self.stride % self.horizon != 0:
                raise ValueError("Stride must be a factor of horizon.")

            # Create a list of instances of PointRegressionForecaster with their step parameter corresponding to the
            # horizon and stride.
            for step in range(stride, horizon+1, stride):
                self.forecasters = []
                self.forecasters.append(PointRegressionForecaster(
                    regressor, step=step, forecast_type=forecast_type, base_forecast_column=base_forecast_column))
            
        def fit(self, X, y):
            for forecaster in self.forecasters:
                forecaster.fit(X, y)
        
        def predict(self, X):
            predictions = [forecaster.predict(X) for forecaster in self.forecasters]
            return pd.concat(predictions, axis=0)