import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit

### Platt Scaling ###
# fit_intercept : boolean
#       True (default) for Platt scaling
#       False          for temperature scaling
# penalty       : str (optional), penalty on the loss function, strength (coefficient) is fixed at 1.0
#       'l1'           for L1 regularization of A (Platt) or 1/T (temperature)
#       'l2'           for L2 regularization
#       'elasticnet'   for both, additively
#       None           for no regularization
def apply_platt_scaling(base_model, X_cal, y_cal):
    platt_calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method='sigmoid')
    platt_calibrated_model.fit(X_cal, y_cal)
    
    calibrator = platt_calibrated_model.calibrated_classifiers_[0].calibrators[0]
    platt_calibrated_model.info = {'A' : -calibrator.a_, 'B' : -calibrator.b_, 'T' : -1/calibrator.a_}

    return platt_calibrated_model

### Isotonic Regression ###
def apply_isotonic_regression(base_model, X_cal, y_cal):
    isotonic_calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method='isotonic') 
    isotonic_calibrated_model.fit(X_cal, y_cal)

    isotonic_calibrated_model.info = {} # consistency
    
    return isotonic_calibrated_model

# Model wrapper for calibration methods taken from the netcal package
class CalibratedBinaryModel(nn.Module):
    """A simple module for histogram binning or beta calibration."""
    def __init__(self, base_model, calibrator, logits = True, info = None):
        super(CalibratedBinaryModel, self).__init__()
        self.base_model = base_model
        self.calibrator = calibrator # from netcal
        self.logits = logits
        self.info = info
        
    def predict_proba(self, X):
        """Return calibrated probability predictions (must be binary)."""
        z = self.base_model.predict_proba(X)
        if self.logits:
            z[:, 1] = torch.logits(z[:, 1])
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
    # I don't expect logit transforming to improve the performance of histogram binning
    hist.fit(y_pred, y_cal)

    info = {'bin_bounds' : hist.get_params()['_bin_bounds'][0],
            'bin_vals'   : hist.get_params()['_bin_map']}
    
    return CalibratedBinaryModel(base_model, hist, logits = False, info = info)

### Beta Calibration (no regularization) ###
# this method explicitly maps probability estimates to calibrated probability estimates so no need to logit
def apply_beta_calibration(base_model, X_cal, y_cal):
    from netcal.scaling import BetaCalibration as BC
    beta = BC("mle")

    y_pred = base_model.predict_proba(X_cal)[:, 1] # probability scale
    beta.fit(y_pred, y_cal)

    tmp = beta.get_params()['_sites']
    a, b = tmp['weights']['values']
    c = tmp['bias']['values'][0]
    info = {'a' : a, 'b' : b, 'c' : c}
    
    return CalibratedBinaryModel(base_model, beta, logits = False, info = info)

### Platt Scaling (no regularization) ###
def apply_platt_scaling_v2(base_model, X_cal, y_cal):
    from netcal.scaling import LogisticCalibration as LC
    platt = LC(temperature_only = False, method = "mle", detection = False)
    
    y_pred = base_model.predict_proba(X_cal)[:, 1] # probability scale
    platt.fit(y_pred, y_cal)

    info = {'A' : platt.weights[0], 'B' : platt.intercept[0]}
    
    return CalibratedBinaryModel(base_model, platt, logits = False, info = info)

### Temperature Scaling (no regularization) ###
def apply_temperature_scaling_v2(base_model, X_cal, y_cal):
    from netcal.scaling import TemperatureScaling as TS
    temp = TS(method = "mle", detection = False)
    
    y_pred = base_model.predict_proba(X_cal)[:, 1] # probability scale
    temp.fit(y_pred, y_cal)

    info = {'T' : 1 / temp.temperature[0]}
    
    return CalibratedBinaryModel(base_model, temp, logits = False, info = info)

###############################################################

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
    
    def softmax(self, Z):
        """Transform scaled logits to output probability predictions."""
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True)) 
        return exps / np.sum(exps, axis=1, keepdims=True) 
        
    def predict_proba(self, X):
        """Return the probability predictions from the scaled logits."""
        scaled_logits = self.predict_scaled_logits(X)
        return self.softmax(scaled_logits)
    
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
