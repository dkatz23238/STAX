from sklearn.base import TransformerMixin
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.stats import boxcox
from scipy.special import inv_boxcox


class BackshiftOperator(TransformerMixin):
    """Backshift operator to transform time series

    fv is first value.
    s is original series.
    ls is lagged series
    """
    def __init__(self):
        self.lags = 1

    def fit(self, X, y=None):
        self.x1 = X[0]
        self.s = X
        self.X_ = shift(X, self.lags, cval=np.nan)
        return self

    def transform(self, X):
        return self.s - self.X_

    def fit_transform(self, X, y=None):
        self.s = X
        self.x1 = X[0]
        self.X_ = shift(X, self.lags, cval=np.nan)
        return self.s - self.X_

    def inverse_transform(self, X):
        X[0] = self.x1
        return X.cumsum()


class BoxCoxTransform(TransformerMixin):
    """Boxcox Transform and reverse
    
    boxcox and reverse boxcox in scikitlearn style


    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X_, lmbda = boxcox(X)
        self.lmbda = lmbda
        self.X_ = X_
        return self

    def transform(self, X, y=None):
        X_ = boxcox(X, lmbda=self.lmbda)
        return X_

    def fit_transform(self, X, y=None):
        X_, lmbda = boxcox(X)
        self.lmbda = lmbda
        self.X_ = X_
        return X_

    def inverse_transform(self, X):
        return inv_boxcox(self.lmbda, X)
