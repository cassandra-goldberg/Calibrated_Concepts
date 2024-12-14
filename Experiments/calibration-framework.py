import torch
from sklearn.calibration import CalibratedClassifierCV
import torch.optim as optim
import torch.nn as nn
import pandas as pd

### Platt Scaling ###
def apply_platt_scaling(base_model, X_cal, y_cal):
    platt_calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    platt_calibrated_model.fit(X_cal, y_cal)
    return platt_calibrated_model

### Isotonic Regression ###
def apply_isotonic_regression(base_model, X_cal, y_cal):
    isotonic_calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit') 
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
        print(original_logits[:5])
        scaled_logits = original_logits / self.temperature
        return scaled_logits
        
    def predict_proba(self, X):
        """Return the probability predictions from the scaled logits."""
        scaled_logits = self.predict_scaled_logits(X)
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)) 
        return exps / np.sum(exps, axis=1, keepdims=True) 
    

def train_temperature_scaling(base_model, X_cal, y_cal):
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
    print(f"Optimal temperature: {temperature_model.temperature:.4f}")
    
    return temperature_model