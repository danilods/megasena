"""Prediction models module."""
import warnings

__all__ = []

# Try to import ARIMA model
try:
    from .arima_model import ARIMAPredictor
    __all__.append('ARIMAPredictor')
except ImportError as e:
    warnings.warn(f"ARIMAPredictor not available: {e}")
    ARIMAPredictor = None

# Try to import Prophet models
try:
    from .prophet_model import ProphetPredictor, MultiTargetProphetPredictor
    __all__.extend(['ProphetPredictor', 'MultiTargetProphetPredictor'])
except ImportError as e:
    warnings.warn(f"Prophet models not available: {e}")
    ProphetPredictor = None
    MultiTargetProphetPredictor = None

# Try to import LSTM model
try:
    from .lstm_model import LSTMPredictor
    __all__.append('LSTMPredictor')
except ImportError as e:
    warnings.warn(f"LSTMPredictor not available: {e}")
    LSTMPredictor = None
