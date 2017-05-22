"""
The :mod:`seqlearn.exceptions` module includes all custom warnings and error
classes used across sequence-learn.
"""

__all__ = ['NotFittedError',
           'FitFailedWarning']


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if the model is used before fitting."""


class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the model."""
