import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

def calibration_error(y, y_prob, measure='K1', bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    y_pred = (y_prob > 0.5).astype(int)
    accuracies = y_pred==y
    calib_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower.item(), y_prob <= bin_upper.item())
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            if measure == 'K1':
                calib_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
            elif measure == 'K2':
                calib_error += (np.abs(avg_confidence_in_bin - accuracy_in_bin)**2) * prob_in_bin
            elif measure == 'Kmax':
                calib_error = max(calib_error, np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin)
    return calib_error

def plot_calibration_curves_cos_sim(models, calibration_metadata_df, calibration_cosine_similarity_df):
    concepts = list(calibration_cosine_similarity_df.columns)
    eces = {}
    fig, ax = plt.subplots(1,1)
    for concept in concepts:
        X_concept = calibration_cosine_similarity_df[[concept]].to_numpy()
        y = (calibration_metadata_df[concept]==1).to_numpy().astype(int)
        y_prob = models[concept].predict_proba(X_concept)[:,1]
        ece = expected_calibration_error(y, y_prob, bins=10)
        eces[concept] = ece
        print(f'Concept: {concept.ljust(10)} | ECE: {ece:.3f}')
        display = CalibrationDisplay.from_predictions(
            y,
            y_prob,
            n_bins=10,
            name=concept,
            ax=ax,
        )
    
    return eces

def plot_calibration_curves_emb(models, calibration_metadata_df, calibration_cosine_similarity_df, 
                                calibration_hidden_states):
    concepts = list(calibration_cosine_similarity_df.columns)
    eces = {}
    fig, ax = plt.subplots(1,1)
    for concept in concepts:
        X_concept = calibration_hidden_states
        y = (calibration_metadata_df[concept]==1).to_numpy().astype(int)
        y_prob = models[concept].predict_proba(X_concept)[:,1]
        ece = expected_calibration_error(y, y_prob, bins=10)
        eces[concept] = ece
        print(f'Concept: {concept.ljust(10)} | ECE: {ece:.3f}')
        display = CalibrationDisplay.from_predictions(
            y,
            y_prob,
            n_bins=10,
            name=concept,
            ax=ax,
        )
    
    return eces




