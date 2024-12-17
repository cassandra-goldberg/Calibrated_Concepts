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

"""
models  : Dictionary of calibration method names (str) to calibrated models
name    : (str) Name of the base model, e.g., M3
concept : For global methods, None. For individual methods, the name of the concept to be plotted
path    : If you want to save the plot to a file, the path to write to. Else, None
"""
def plot_calibrator(models, name, concept = None, path = None):
    from scipy.special import logit, expit
    fig, ax = plt.subplots()
    n = 501 # discretization
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # None
    plt.plot([0, 1], [0, 1], label = 'None', color = 'grey', linestyle = 'dashed', alpha = 0.5)

    for method, model in models:
        if not concept is None:
            model = model[concept]
        if method == 'Histogram': # netcal implementation
            x_vals = model.calibrator.get_params()['_bin_bounds'][0]
            y_vals = model.calibrator.get_params()['_bin_map']
            y_vals = np.append(y_vals, y_vals[-1])
            plt.step(x_vals, y_vals, where = 'post', label = 'Histogram', color = colors[0])
        elif method == 'Isotonic': # sklearn implementation
            x_vals = np.linspace(0, 1, num=n, endpoint=True)[1:-1]
            y_vals = model.calibrated_classifiers_[0].calibrators[0].predict(logit(x_vals))
            plt.plot(x_vals, y_vals, label = 'Isotonic', color = colors[1])
        elif method == 'Platt': # sklearn implementation
            calibrator = model.calibrated_classifiers_[0].calibrators[0]
            a, b = calibrator.a_, calibrator.b_
            
            x_vals = np.linspace(0, 1, num=n, endpoint=True)[1:-1]
            y_vals = calibrator.predict(logit(x_vals))
            plt.plot(x_vals, y_vals, label = 'Platt (A={:.2f}, B={:.2f})'.format(a, b), color = colors[2])
        elif method == 'Temperature' : # manual implementation, FIXME weird
            T = model.temperature
            x_vals_vec = np.array([1 - x_vals, x_vals]).T
            y_vals = model.softmax(x_vals_vec / T)[:, 1]
            plt.plot(x_vals, y_vals, label = 'Temperature (T={:.2f})'.format(T), color = colors[3])
        elif method == 'Temperature v2': # bastardized interpretation of manual implementation
            x_vals = np.linspace(0, 1, num=n, endpoint=True)[1:-1]
            y_vals = expit(logit(x_vals) / model.temperature)
            plt.plot(x_vals, y_vals, label = 'Temperature (T={:.2f})'.format(model.temperature), color = colors[3])
        elif method == 'Temperature v3': # netcal implementation
            T = model.calibrator.temperature() # probably need to index
            x_vals = np.linspace(0, 1, num=n, endpoint=True)
            y_vals = model.calibrator.transform(x_vals)
            plt.plot(x_vals, y_vals, label = 'Temperature (T={:.2f})'.format(T), color = colors[3])
        elif method == 'Beta': # netcal implementation
            tmp = model.calibrator.get_params()['_sites']
            a, b = tmp['weights']['values']
            c = tmp['bias']['values'][0]

            x_vals = np.linspace(0, 1, num=n, endpoint=True)
            y_vals = model.calibrator.transform(x_vals)
            plt.plot(x_vals, y_vals, label = 'Beta (a={:.2f}, b={:.2f}, c={:.2f})'.format(a, b, c), color = colors[4])
        else:
            print("{} plot not yet implemented".format(method))
            continue
    
    plt.legend()
    ax.set_xlabel('Base model probability estimate')
    ax.set_ylabel('Calibrated model probability estimate')
    ax.set_title('Calibrators of {} Model'.format(name))

    if not path is None:
        plt.savefig(path)
    
    plt.show()

