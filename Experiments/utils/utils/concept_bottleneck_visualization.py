'''
2. Concept Model Utils
    d) Visualization
'''
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from matplotlib import pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_map = {'Base' : 'grey', 
             'Histogram' : colors[0], 
             'Isotonic' : colors[1], 
             'Platt L2' : colors[2],          # regularized (sklearn)
             'Temperature' : colors[3],       # unregularized (netcal)
             'Beta' : colors[4], 
             'Platt' : colors[5],             # unregularized (netcal)
             'Temperature L2' : colors[6],    # regularized (sklearn)
             'Platt L1' : colors[8]           # regularized (sklearn)
            }

def plot_calibrators(models, name, concept = None, path = None, methods = None):
    """
    Plot the calibration functions produced by several calibration methods against each other with one
    curve for each method (for per-concept calibration functions, a concept must be specified)
    === Parameters ===
    models  : Dictionary of calibration method names (str) to calibrated models
    name    : (str) Name of the base model, e.g., M3
    concept : For global methods, None. For individual methods, the name of the concept to be plotted
    path    : If you want to save the plot to a file, the path to write to. Else, None
    methods : If you only want to plot a strict subset of the calibrators, list the names. Else, None
    """
    
    fig, ax = plt.subplots()
    n = 501 # discretization

    if methods is None:
        methods = list(models.keys())
        methods.remove('Base')
    if concept is None:
        methods = [m for m in methods if not m.endswith('individual')]

    # None
    plt.plot([0, 1], [0, 1], label = 'None', color = color_map['Base'], linestyle = 'dashed', alpha = 0.5)

    for method, model in models.items():
        if not method in methods:
            continue

        method = method.split(" ")
        method, version = " ".join(method[0:-1]), method[-1]
        
        if method == 'Histogram':
            x_vals = model.params['bin_bounds']
            y_vals = model.params['bin_vals']
            y_vals = np.append(y_vals, y_vals[-1])
            if version == 'global':
                plt.step(x_vals, y_vals, where = 'post', label = 'Histogram', color = color_map[method])
            elif version == 'individual':
                plt.step(x_vals, y_vals, where = 'post', label = 'Histogram', color = color_map[method], marker = '.')
            
        else:
            if version == 'individual':
                calibrator = model.calibrators[concept]
            elif version == 'global':
                calibrator = model.calibrator
            else:
                raise ValueError('Unrecognized calibration method {} is not global or individual'.format(method))
            x_vals = np.linspace(0, 1, num=n, endpoint=True)[1:-1]
            if model.logits:
                z_vals = logit(x_vals)
            else:
                z_vals = x_vals
            y_vals = model.transform(calibrator, z_vals)
            label = ", ".join("{}={:.2f}".format(k, v) for k, v in model.params.items())
            label = "{} ({})".format(method, label) if len(label) > 0 else method
            if version == 'global':
                plt.plot(x_vals, y_vals, label = label, color = color_map[method])
            elif version == 'individual':
                plt.plot(x_vals, y_vals, label = label, color = color_map[method], marker = '.')
    
    plt.legend()
    ax.set_xlabel('Base model probability estimate')
    ax.set_ylabel('Calibrated model probability estimate')
    ax.set_title('Calibrators of {} Model'.format(name))

    if not path is None:
        plt.savefig(path)
    
    plt.show()
    return


def compute_calibrator_magnitudes(model, concepts = None, n = 101, return_cov = False):
    """
    Get calibrators' distance from the identity function, using Mahalanobis distance
    === Parameters ===
    model    : Calibrated model with individual-type calibration method
    concepts : (optional list) Subset of the concepts to be plotted. If None, all are plotted
    n        : (optional int) Granularity with which to sample the calibration functions along the unit interval
    """
    if concepts is None:
        concepts = list(model.concepts)

    x_vals = np.linspace(0, 1, num=n, endpoint=True)[1:-1]
    mtx = np.empty(shape=(len(concepts), len(x_vals)))

    # 1. get the covariance matrix
    for i, concept in enumerate(concepts):
        calibrator = model.calibrators[concept]
        
        if model.logits:
            z_vals = logit(x_vals)
        else:
            z_vals = x_vals
        
        mtx[i, :] = model.transform(calibrator, z_vals)

    cov_mtx = np.cov(mtx.T) # cov() takes input where cols are obs (concepts), rows are vars (pts on unit interval)

    # 2. it's usually sort of degenerate, so get the utilities to project each calibration function into a lower-dimensional subspace
    # get evecs for evals within some factor of the maximum
    evals, evecs = np.linalg.eigh(cov_mtx)
    j = evals / evals[-1] > 1e-6
    ebasis, scale = evecs[:, j], evals[j]**0.5
    # center the variables to have mean zero
    mu = np.mean(mtx, axis = 0)
    mtx = mtx - mu.reshape(1, -1)
    # center, project, and scale the identity function
    u = ebasis @ (np.dot(ebasis.T, x_vals - mu) * scale)

    # 3. get the distances
    dists = {}
    for i, concept in enumerate(concepts):
        # project onto the subspace, scaled by square root of corresponding eigenvalue
        v = ebasis @ ((ebasis.T @ mtx[i, :]) * scale)
        # use Euclidean distance
        dists[concept] = ((v - u)**2).sum()**0.5
    
    if return_cov:
        return dists, cov_mtx
    return dists


def plot_individual_calibrator(model, name, method, concepts = None, path = None, top_k = 10, color_threshold = None, size = 5):
    """
    Plot the calibration functions produced by several calibration methods against each other with one
    curve for each method (for per-concept calibration functions, a concept must be specified)
    === Parameters ===
    model           : Calibrated model with individual-type calibration method
    name            : (str) Name of the base model, e.g., 'M3'
    method          : (str) Name of the calibration method, e.g., 'Platt L2'
    concepts        : (optional list) Subset of the concepts to be plotted. If None, all are plotted
    path            : (optional str) If you want to save the plot to a file, the path to write to. Else, None
    top_k           : (optional int) Number of concepts to be labeled, selected by maximum Mahalanobis distance from the identity
    color_threshold : (optional int) Threshold for Mahalonobis distance from the identity above which a concept is labeled
    size            : (int) Size of the output plot along each axis
    """
    fig, ax = plt.subplots(figsize=(1.5*size, size))
    n = 501 # discretization

    if concepts is None:
        concepts = list(model.concepts)

    dists = compute_calibrator_magnitudes(model, concepts)
    max_dist = np.max([v for _, v in dists.items()])
    if not color_threshold is None:
        labeled_curves = [k for k, v in dists.items() if v > color_threshold]
    elif not top_k is None:
        labeled_curves = pd.Series(dists).sort_values(ascending=False).index[:top_k]
    else:
        raise ValueError('Parameters top_k and color_threshold cannot both be None')

    # None
    plt.plot([0, 1], [0, 1], label = 'y=x', color = color_map['Base'], linestyle = 'dashed')

    color_idx = 0
    for concept in concepts:
        args = {}
        if concept in labeled_curves:
            # set: label, color, linestyle
            label = ', '.join('{}={:.2f}'.format(k, v) for k, v in model.params[concept].items())
            args['label'] = '{} ({})'.format(concept, label) if len(label) > 0 else concept
            if colors[color_idx] == color_map[method]:
                color_idx = (color_idx + 1) % len(colors)
            args['color'] = colors[color_idx]
            color_idx = (color_idx + 1) % len(colors)
            args['linestyle'] = 'dotted'
            args['linewidth'] = 3
        else:
            # set: color, alpha
            args['color'] = color_map[method]
            args['alpha'] = 0.01 + (0.25 * dists[concept] / max_dist)

        if method == 'Histogram':
            x_vals = model.params['bin_bounds']
            y_vals = model.params['bin_vals']
            y_vals = np.append(y_vals, y_vals[-1])
            plot_fn = plt.step
        else:
            calibrator = model.calibrators[concept]
            
            x_vals = np.linspace(0, 1, num=n, endpoint=True)[1:-1]
            if model.logits:
                z_vals = logit(x_vals)
            else:
                z_vals = x_vals

            y_vals = model.transform(calibrator, z_vals)
            plot_fn = plt.plot

        plot_fn(x_vals, y_vals, **args)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #plt.legend()
    ax.set_xlabel('Base model probability estimate')
    ax.set_ylabel('Calibrated model probability estimate')
    ax.set_title('{} Calibrators of {} Model'.format(method, name))

    if not path is None:
        plt.savefig(path)
    
    plt.show()
    return dists


# FIXME: why do the calibration error utilities inline calibration errror measurement instead 
# of invoking a function from utils.py? Also, which calibration error metric is used?

def plot_calibration_errors(concept_models, testX, testy, name, path = None, methods = None, nbins = 15):
    """
    Plot the calibration errors produced by several calibration methods against each other with one
    curve for each method. IMPORTANT: This pools all concepts into a single curve.
    === Parameters ===
    concept_models : Dictionary of calibration method names (str) to calibrated models
    testX          : Input dataset to the concept model
    testy          : Corresponding concept labels
    name           : (str) Name of the base model, e.g., M3
    concept        : (optional str) For global methods, None. For individual methods, the name of the concept to be plotted
    path           : (optional str) If you want to save the plot to a file, the path to write to. Else, None
    methods        : (optional list) If you only want to plot a strict subset of the calibrators, list the names. Else, None
    nbins          : (int) Number of bins to cut the sample into for level sets
    """
    if methods is None:
        methods = concept_models.keys()
        
    # None
    plt.plot([0, 1], [0, 1], label = 'y=x', color = 'black', linestyle = 'dashed')
    
    for method in methods:
        y_vals = concept_models[method].predict_proba(testX)
        probs = pd.DataFrame.from_dict({'model_prob' : y_vals.flatten(), 'true_label' : testy.flatten()})
        #probs = probs.sort_values(by='model_prob').reset_index(drop=True)
        probs['level_set'] = pd.cut(probs.model_prob, bins=nbins, include_lowest=True)
        probs = probs.groupby(by='level_set', observed=True).agg(model_prob=('model_prob', 'mean'), 
                                                  true_prob=('true_label', 'mean')) #, bin_size=('model_prob', 'count'))
        probs = probs.sort_values(by='true_prob').reset_index(drop=True)
        #probs['bin_prob'] = probs['bin_size'] / probs['bin_size'].sum()
        if method == 'Base':
            k = method
            linestyle = 'solid'
        else:
            k = method.rsplit(' ', 1)[0]
            linestyle = 'solid' if method.rsplit(' ', 1)[-1].lower() == 'global' else 'dotted'
            linewidth = 1 if linestyle == 'solid' else 2
        plt.plot(probs['true_prob'], probs['model_prob'], label = method, color = color_map[k], 
                 marker = 'o', linestyle = linestyle, linewidth = linewidth)

    plt.title('Calibration error curves of {} Model'.format(name))
    plt.xlabel('Empirical probability of concept')
    plt.ylabel('Model predicted probability of concept')
    plt.legend()

    if not path is None:
        plt.savefig(path)
        
    plt.show()
    return


def plot_concept_calibration_errors(model, testX, testy, name, method, path = None, concepts = None, nbins = 15):
    """
    Plot the calibration errors produced by a single calibration method for different concepts against each other
    with one curve for each concept.
    === Parameters ===
    model    : Individually calibrated model
    testX    : Input dataset to the concept model
    testy    : Corresponding concept labels
    name     : (str) Name of the base model, e.g., 'M3'
    method   : (str) Name of the calibration method, e.g., 'Platt L2'
    path     : (optional str) If you want to save the plot to a file, the path to write to. Else, None
    concepts : (optional list) If you only want to plot a strict subset of the calibrators, list the concepts. Else, None
    nbins    : (int) Number of bins to cut the sample into for level sets
    """
    if concepts is None:
        concepts = model.concepts
        
    # None
    plt.plot([0, 1], [0, 1], label = 'y=x', color = 'black', linestyle = 'dashed')
    
    y_vals = model.predict_proba(testX)
    
    for i, concept in enumerate(concepts):
        probs = pd.DataFrame.from_dict({'model_prob' : y_vals[:, i], 'true_label' : testy[:, i]})
        probs['level_set'] = pd.cut(probs.model_prob, bins=nbins, include_lowest=True)
        probs = probs.groupby(by='level_set', observed=True).agg(model_prob=('model_prob', 'mean'), 
                                                  true_prob=('true_label', 'mean'))
        probs = probs.sort_values(by='true_prob').reset_index(drop=True)

        args = {}
        args['label'] = concept
        args['color'] = color_map[method]
        args['marker'] = 'o'
        args['alpha'] = 0.1
        
        plt.plot(probs['model_prob'], probs['true_prob'], **args)

    plt.title('{} calibration error curves of {} Model'.format(method, name))
    plt.xlabel('Model predicted probability of concept')
    plt.ylabel('Empirical probability of concept')
    #plt.legend()

    if not path is None:
        plt.savefig(path)
        
    plt.show()
    return


# 6. Evaluate > Summarize individual calibrators
def get_calibrator_params(concept_models):
    temp_Ts  = np.array([params['T'] for params in concept_models['Temperature individual'].params.values()])
    platt_As = np.array([params['A'] for params in concept_models['Platt individual'].params.values()])
    platt_Bs = np.array([params['B'] for params in concept_models['Platt individual'].params.values()])
    beta_as  = np.array([params['a'] for params in concept_models['Beta individual'].params.values()])
    beta_bs  = np.array([params['b'] for params in concept_models['Beta individual'].params.values()])
    beta_cs  = np.array([params['c'] for params in concept_models['Beta individual'].params.values()])

    params = {'Temperature (T)' : temp_Ts, 
              'Platt (A)'       : platt_As, 
              'Platt (B)'       : platt_Bs, 
              'Beta (a)'        : beta_as, 
              'Beta (b)'        : beta_bs, 
              'Beta (c)'        : beta_cs
             }
    return params


def print_calibrator_params(params):
    for name, arr in params.items():
        est = np.mean(arr)
        n = len(arr)
        ste = ((arr - est)**2).sum()
        ste = ste**0.5 / (n * (n - 1))**0.5
        print("{} {} : {:.3f} \u00B1 {:.3f}".format(name, " "*(16 - len(name)), est, ste))
    return
