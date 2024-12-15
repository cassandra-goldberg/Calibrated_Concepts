import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay

def add_split_column(df):
    # Generate a random assignment for each row
    np.random.seed(42)
    random_values = np.random.choice(
        ['train', 'test', 'calibration'],
        size=len(df),
        p=[0.6, 0.2, 0.2]  # Probabilities for train, test, and calibration
    )
    # Add the new 'split' column to the DataFrame
    df['split'] = random_values
    return df

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

def get_test_classification_metric(models, metadata_df, cosine_similarity_df,
                                   embeddings, input_type='similarity',
                                   metric='accuracy'):
    concepts = list(cosine_similarity_df.columns)
    values = {}
    model = models
    for concept in concepts:
        if input_type == 'similarity':
            X = cosine_similarity_df[[concept]].to_numpy()
        elif input_type == 'embeddings':
            X = embeddings
        y = (metadata_df[concept]==1).to_numpy().astype(int)
        if type(models) == dict:
            model = models[concept]

        if metric == 'accuracy':
            y_pred = model.predict(X)
            values[concept] = accuracy_score(y, y_pred)
        elif metric == 'f1':
            y_pred = model.predict(X)
            values[concept] = f1_score(y, y_pred)
        elif metric == 'auc':
            y_score = model.predict_proba(X)[:,1]
            values[concept] = roc_auc_score(y, y_score)
        elif 'K' in metric:
            y_score = model.predict_proba(X)[:,1]
            values[concept] = calibration_error(y, y_score, measure=metric)
    return values

def get_all_models_classification_metric(base_models, metadata_df, cosine_similarity_df,
                                         embeddings, metric='accuracy', input_type='similarity'):
    model_names = base_models.keys()
    values_list = []
    for model_name in model_names:
        if 'M5' in model_name:
            values = get_test_classification_metric(base_models[model_name], metadata_df, cosine_similarity_df,
                           embeddings, input_type='embeddings', metric=metric)
        elif 'Threshold' in model_name and (metric=='auc' or 'K' in metric):
            concepts = list(cosine_similarity_df.columns)
            values = dict.fromkeys(concepts, '-')
        else:
            values = get_test_classification_metric(base_models[model_name], metadata_df, cosine_similarity_df,
                           embeddings, input_type=input_type, metric=metric)
        values['Model'] = model_name
        values_list.append(values)
    comparison_df = pd.DataFrame.from_dict(values_list)
    comparison_df = comparison_df.set_index('Model')
    return comparison_df

def compare_all_models_calibration_metric(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings, 
                                   metric='accuracy'):
    base_models_df = get_all_models_classification_metric(base_models, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric)
    base_models_df = base_models_df.reset_index()
    base_models_df['Calibration'] = 'None'
    
    cal_m3_models_df = get_all_models_classification_metric(m3_models_cal, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric)
    cal_m3_models_df = cal_m3_models_df.reset_index()
    cal_m3_models_df['Calibration'] = cal_m3_models_df['Model']
    cal_m3_models_df['Model'] = '(M3) Global Similarity LogReg'
    
    cal_m4_models_df = get_all_models_classification_metric(m4_models_cal, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric)
    cal_m4_models_df = cal_m4_models_df.reset_index()
    cal_m4_models_df['Calibration'] = cal_m4_models_df['Model']
    cal_m4_models_df['Model'] = '(M4) Individual Similarity LogReg'
    
    cal_m5_models_df = get_all_models_classification_metric(m5_models_cal, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric, input_type='embeddings')
    cal_m5_models_df = cal_m5_models_df.reset_index()
    cal_m5_models_df['Calibration'] = cal_m5_models_df['Model']
    cal_m5_models_df['Model'] = '(M5) Embeddings LogReg'

    df = pd.concat([base_models_df, cal_m3_models_df, cal_m4_models_df, cal_m5_models_df])
    df = df.sort_values(by=['Model'])
    df = df.set_index(['Model','Calibration'])
    df = df.round(3)
    return df

def compare_all_models_calibration_avg(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings):
    metrics_df = pd.DataFrame()
    for metric in ['accuracy', 'f1', 'auc', 'K1', 'K2', 'Kmax']:
        df = compare_all_models_calibration_metric(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings, 
                                   metric=metric)
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace('-', np.nan).infer_objects(copy=False)
        series_mean = df.transpose().mean(skipna=True)
        series_std = df.transpose().std(skipna=True)
        series_str = series_mean.apply(lambda x: '{0:.3f}'.format(x))+u" \u00B1 "+series_std.apply(lambda x: '{0:.2f}'.format(x))
        metrics_df[metric] = series_str
    metrics_df = metrics_df.replace(u"nan \u00B1 nan", '-')
    return metrics_df

def compare_all_models_calibration_concept(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                           test_metadata_df, test_cosine_similarity_df, test_embeddings,
                                          concept):
    metrics_df = pd.DataFrame()
    for metric in ['accuracy', 'f1', 'auc', 'K1', 'K2', 'Kmax']:
        df = compare_all_models_calibration_metric(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings, 
                                   metric=metric)
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace('-', np.nan).infer_objects(copy=False)
        series = df[concept]
        metrics_df[metric] = series.apply(lambda x: '{0:.3f}'.format(x))
    metrics_df = metrics_df.replace('nan', '-')
    return metrics_df