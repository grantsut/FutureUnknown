# Initial file for tests. Will be split as the number of tests grows.

import sys
import pytest
from sklearn.linear_model import LinearRegression 
from futureunknown.regression import ForecasterMixin, PointRegressionForecaster


class Test_index_validator():
    """ Test the _validate_clean_index method of the ForecasterMixin class with various types of good and bad indices. """

    def test_good_daily_index(self):
        """ Test a good daily index. """
        index = pd.date_range(start="2019-01-01", end="2019-01-31", freq="D")
        assert ForecasterMixin()._validate_clean_index(index) == index
    
    def test_good_monthly_index(self):
        """ Test a good monthly index. """
        index = pd.date_range(start="2019-01-01", end="2019-12-31", freq="M")
        assert ForecasterMixin()._validate_clean_index(index) == index
    
    def test_good_business_monthly_index(self):
        """ Test a good business monthly index. """
        index = pd.date_range(start="2019-01-01", end="2019-12-31", freq="BM")
        assert ForecasterMixin()._validate_clean_index(index) == index
    
    def test_bad_daily_index(self):
        """ Test a daily index in which dates are missing. """
        index = pd.date_range(start="2019-01-01", end="2019-01-31", freq="D")
        index = index.drop(index[3])
        with pytest.raises(ValueError):
            ForecasterMixin()._validate_clean_index(index)
    
    def test_non_datetime_index(self):
        """ Test an index that is not a DatetimeIndex. """
        index = pd.Index([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            ForecasterMixin()._validate_clean_index(index)
    
    def test_non_monotonic_index(self):
        """ Test that an incorrectly-sorted index is fixed. """
        index = pd.date_range(start="2019-01-01", end="2019-01-31", freq="D")
        index = index[::-1]
        index = ForecasterMixin()._validate_clean_index(index)
        assert index.is_monotonic_increasing
    