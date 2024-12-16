import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import minimize

### Platt Scaling ###
def apply_platt_scaling(base_model, X_cal, y_cal):
    platt_calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method='sigmoid')
    platt_calibrated_model.fit(X_cal, y_cal)
    return platt_calibrated_model

### Isotonic Regression ###
def apply_isotonic_regression(base_model, X_cal, y_cal):
    isotonic_calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method='isotonic') 
    isotonic_calibrated_model.fit(X_cal, y_cal)
    return isotonic_calibrated_model

### Temperature Scaling ###
class TemperatureScaling(nn.Module):
    """A simple module for temperature scaling."""
    def __init__(self, base_model):
        super(TemperatureScaling, self).__init__()
        self.base_model = base_model
        
    def predict_scaled_logits(self, X):
        """Scale the logits of the base model using the current temperature."""
        original_logits = self.base_model.predict_proba(X) 
        scaled_logits = original_logits / self.temperature
        return scaled_logits
        
    def predict_proba(self, X):
        """Return the probability predictions from the scaled logits."""
        scaled_logits = self.predict_scaled_logits(X)
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)) 
        return exps / np.sum(exps, axis=1, keepdims=True) 
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
        
    

def apply_temperature_scaling(base_model, X_cal, y_cal, verbose=True):
    """Train temperature scaling using negative log-likelihood. """
    
    temperature_model = TemperatureScaling(base_model)  # Initialize the temperature scaling model
    
    def nll_loss(temperature):
        """Negative log-likelihood loss for the given temperature."""
        temperature_model.temperature = temperature[0]  # Update temperature
        scaled_logits = temperature_model.predict_scaled_logits(X_cal) 
        # Compute cross-entropy loss
        log_probs = scaled_logits - np.log(np.sum(np.exp(scaled_logits), axis=1, keepdims=True))
        nll = -np.mean(log_probs[np.arange(len(y_cal)), y_cal]) 
        return nll
    
    # Minimize the negative log-likelihood loss to find the optimal temperature
    result = minimize(nll_loss, x0=[1], bounds=[(1e-2, 10.0)], tol=1, method='L-BFGS-B')  # Temperature > 0
    
    # Store the optimal temperature
    temperature_model.temperature = result.x[0]
    if verbose:
        print(f"Optimal temperature: {temperature_model.temperature:.4f}")
    
    return temperature_model

# Model wrapper for calibration methods taken from the netcal package
class CalibratedBinaryModel(nn.Module):
    """A simple module for histogram binning or beta calibration."""
    def __init__(self, base_model, calibrator):
        super(CalibratedBinaryModel, self).__init__()
        self.base_model = base_model
        self.calibrator = calibrator # from netcal
        
    def predict_proba(self, X):
        """Return calibrated probability predictions (must be binary)."""
        z = self.base_model.predict_proba(X)
        z[:, 1] = self.calibrator.transform(z[:, 1])
        z[:, 0] = 1 - z[:, 1]
        return z
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

### Histogram Binning ###
def apply_histogram_binning(base_model, X_cal, y_cal, nbins = 10):
    from netcal.binning import HistogramBinning as HB
    hist = HB(nbins, detection = False) #, equal_intervals = False is not implemented :(
    
    y_pred = base_model.predict_proba(X_cal)[:, 1] # estimated probability that the (concept) label is 1
    hist.fit(y_pred, y_cal)
    
    return CalibratedBinaryModel(base_model, hist)

### Beta Calibration ###
def apply_beta_calibration(base_model, X_cal, y_cal):
    from netcal.scaling import BetaCalibration as BC
    beta = BC("mle")

    y_pred = base_model.predict_proba(X_cal)[:, 1]
    beta.fit(y_pred, y_cal)
    
    return CalibratedBinaryModel(base_model, beta)
