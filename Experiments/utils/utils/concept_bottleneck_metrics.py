'''
2. Concept Model Utils
    a) Metrics
'''
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from .utils import calibration_error

def get_binomial_ste(mu, n):
    return (mu * (1 - mu) / n)**0.5
    
def get_normal_ste(a1, a2, n):
    ste = ((a1 - a2)**2).sum()
    return ste**0.5 / (n * (n - 1))**0.5

def concept_accuracies(all_y_pred, all_y_true, concepts, verbose = True, ste_coef=1, individual = True):
    ret = {}
    
    # individual concept accuracies
    if individual:
        for i, concept in enumerate(concepts):
            y_true = all_y_true[:, i]
            y_pred = all_y_pred[:, i]
            
            accuracy = accuracy_score(y_true, y_pred)
            #ste = ste_coef * (accuracy * (1 - accuracy) / len(y_true))**0.5
            ste = ste_coef * get_binomial_ste(accuracy, len(y_true))
            ret[concept] = (accuracy, ste)
            
            if verbose:
                print(f"{i}. {concept} Test Accuracy: {accuracy * 100:.2f} \u00B1 {ste * 100:.2f}%")

    # overall accuracy
    accuracy = accuracy_score(all_y_true.flatten(), all_y_pred.flatten())
    ste = ste_coef * get_binomial_ste(accuracy, len(all_y_true.flatten()))

    if verbose:
        print(f"Test Accuracy: {accuracy * 100:.2f} \u00B1 {ste * 100:.2f}%")

    if individual:
        return accuracy, ste, ret
    return accuracy, ste, None

def concept_calibration_errors(all_y_pred, all_y_true, concepts, verbose = True, 
                               measure = 'K1', bins=10, individual = True):
    ret = {}
    
    # individual concept calibration errors
    if individual:
        for i, concept in enumerate(concepts):
            y_true = all_y_true[:, i]
            y_pred = all_y_pred[:, i]
            
            cal_error = calibration_error(y_true, y_pred, measure, bins)
            ret[concept] = cal_error
            
            if verbose:
                print(f"{i}. {concept} Test {measure}-Calibration Error: {cal_error:.4f}")

    # overall calibration error
    cal_error = calibration_error(all_y_true.flatten(), all_y_pred.flatten(), measure=measure, bins=bins)

    if verbose:
        print(f"Test {measure}-Calibration Error: {cal_error:.4f}")

    if individual:
        return cal_error, ret
    return cal_error, None

def target_accuracy(y_pred, y_true, verbose = True, ste_coef=1):
    accuracy = accuracy_score(y_true, y_pred)
    ste = ste_coef * get_binomial_ste(accuracy, len(y_true))

    if verbose:
        print(f"Test Accuracy: {accuracy * 100:.2f} \u00B1 {ste * 100:.2f}%")
    
    return accuracy, ste

'''
6. Evaluate
'''

def train_cal_test_accuracy(concept_model, seq_model, trainX, trainy, calX, caly, testX, testy):
    '''
    Base model main task accuracy by dataset
    '''
    acc, ste = target_accuracy(seq_model.predict(concept_model.decision_function(trainX)), trainy, verbose = False, ste_coef = 2)
    print("Train accuracy        : {:.2f} \u00B1 {:.2f}%".format(acc * 100, ste * 100))
    acc, ste = target_accuracy(seq_model.predict(concept_model.decision_function(calX)), caly, verbose = False, ste_coef = 2)
    print("Calibration accuracy  : {:.2f} \u00B1 {:.2f}%".format(acc * 100, ste * 100))
    acc, ste = target_accuracy(seq_model.predict(concept_model.decision_function(testX)), testy, verbose = False, ste_coef = 2)
    print("Test accuracy         : {:.2f} \u00B1 {:.2f}%".format(acc * 100, ste * 100))

def main_task_test_accuracy(concept_models, seq_models, testX, testy, metrics_dict):
    '''
    Main task test accuracy by calibrator
    '''
    name = "Calibration method"
    print("{} {} Accuracy (%)".format(name, " "*(26 - len(name))))
    
    col = {}
    
    for name in seq_models.keys():
        test_concepts_pred = concept_models[name].decision_function(testX)
        test_target_pred   = seq_models[name].predict(test_concepts_pred)
        
        seq_acc, acc_ste = target_accuracy(test_target_pred, testy.to_numpy(), verbose = False, ste_coef = 2)
        vstr = "{:.2f} \u00B1 {:.2f}".format(seq_acc * 100, acc_ste * 100)
        
        print("{} {} {}".format(name, " "*(26 - len(name)), vstr))
        col[name] = vstr
    
    metrics_dict['Main task accuracy (%)'] = pd.Series(col)


def concept_test_accuracy(concept_models, testX, testy, concepts, metrics_dict):
    '''
    Concept test accuracy by calibrator
    '''
    name = "Calibration method"
    print("{} {} Concept accuracy (%)".format(name, " "*(26 - len(name))))
    
    col = {}
    
    for name, model in concept_models.items():
        acc, acc_ste, _ = concept_accuracies(
            model.predict(testX), testy.numpy(), concepts=concepts, 
            verbose = False, individual = False, ste_coef = 2)
        vstr = "{:.2f} \u00B1 {:.2f}".format(acc * 100, acc_ste * 100)
        print("{} {} {}".format(name, " "*(26 - len(name)), vstr))
        col[name] = vstr
    
    metrics_dict['Concept accuracy (%)'] = pd.Series(col)

'''
End task calibration error by calibrator
'''
def calibration_error_str(y_true, y_pred, measure = 'K1'):
    est = calibration_error(y_true, y_pred, measure = measure)
    return "{:.3f}".format(est)
    #ste = get_ste(est, len(y_pred))
    #vstr = "{:.3f} \u00B1 {:.3f}".format(est * 100, 2 * ste * 100)
    #return vstr

def _main_task_test_calibration_error(name, y_true, y_pred):
    k1_vstr   =  calibration_error_str(y_true, y_pred, 'K1')
    k2_vstr   =  calibration_error_str(y_true, y_pred, 'K2')
    kmax_vstr =  calibration_error_str(y_true, y_pred, 'Kmax')
    return k1_vstr, k2_vstr, kmax_vstr

def main_task_test_calibration_error(concept_models, seq_models, testX, testy, metrics_dict):
    name = "Calibration method"
    print("{} {} K1       K2       Kmax".format(name, " "*(26 - len(name))))
    
    k1_col   = {}
    k2_col   = {}
    kmax_col = {}
    
    for name in seq_models.keys():
        test_concepts_pred = concept_models[name].decision_function(testX)
        test_target_pred = seq_models[name].predict_proba(test_concepts_pred)

        k1_vstr, k2_vstr, kmax_vstr = _main_task_test_calibration_error(name, testy, test_target_pred)
        
        k1_col[name]   = k1_vstr
        k2_col[name]   = k2_vstr
        kmax_col[name] = kmax_vstr
        
        print("{} {} {}    {}    {}".format(
            name, " "*(26 - len(name)), k1_vstr, k2_vstr, kmax_vstr))
        
    metrics_dict['Main task K1']   = pd.Series(k1_col)
    metrics_dict['Main task K2']   = pd.Series(k2_col)
    metrics_dict['Main task Kmax'] = pd.Series(kmax_col)

def create_metrics_dataframe(metrics_dict):
    '''
    Output table
    '''
    da = pd.DataFrame.from_dict(metrics_dict)
    idx = da.index.str.title().str.replace('Base', 'None -').str.rsplit(" ", n=1)
    idx = pd.MultiIndex.from_tuples([tuple(i) for i in idx]) #, names=["Calibration", "Level"]
    da.index = idx
    
    with pd.option_context("future.no_silent_downcasting", True):
        da = da.replace(np.nan, '-').infer_objects(copy=False)
    
    return da

if __name__ == '__main__':
    print('hello world')

    import pickle
    import torch
    from .concept_bottleneck_models import *
    dataset_name = 'CUB'

    metadata_df = pd.read_csv(f'../../Data/{dataset_name}/metadata.csv')
    cosine_similarity_df = pd.read_csv(f'../Cosine_Similarities/{dataset_name}/cosine_similarities.csv')

    concepts = list(cosine_similarity_df.columns)
    test_mask = metadata_df['split'] == 'test'

    test_cosine_similarity_df = cosine_similarity_df[test_mask].reset_index(drop=True)
    test_metadata_df = metadata_df[test_mask].reset_index(drop=True)
    test_concepts = torch.from_numpy(test_metadata_df[concepts].to_numpy())

    with open('../Models/CBM/M1/concepts-base-2.pkl', 'rb') as f:
        M1 = pickle.load(f)
    M1_test_concepts_pred = M1.predict(test_cosine_similarity_df)
    cal_error_k1, k1_errors = concept_calibration_errors(
        M1_test_concepts_pred, test_concepts.numpy(), concepts = concepts, 
        verbose = True, measure = 'K1', bins=10, individual=False)
    cal_error_kmax, kmax_errors = concept_calibration_errors(
        M1_test_concepts_pred, test_concepts.numpy(), concepts = concepts, 
        verbose = True, measure = 'Kmax', bins=10, individual=False)