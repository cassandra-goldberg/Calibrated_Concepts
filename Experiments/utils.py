import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns

models_order = ['GT','CT','GLR','CLR','EmbCLR']
calibration_ordering = ["None", "Histogram", "Isotonic", "Platt", "Temperature", "Beta"]

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

def calibration_error(y_true, y_prob, measure='K1', bins=10):
    bin_boundaries = np.linspace(0, 1.00001, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calib_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob > bin_lower.item(), y_prob <= bin_upper.item())
        prob_in_bin = in_bin.mean()
        if prob_in_bin.item() > 0:
            fract_positives = y_true[in_bin].mean()
            mean_prob = y_prob[in_bin].mean()
            if measure == 'K1':
                calib_error += np.abs(mean_prob - fract_positives) * prob_in_bin
            elif measure == 'K2':
                calib_error += (np.abs(mean_prob - fract_positives)**2) * prob_in_bin
            elif measure == 'Kmax':
                calib_error = max(calib_error, np.abs(mean_prob - fract_positives) * prob_in_bin)
    return calib_error

def get_test_classification_metric(models, metadata_df, cosine_similarity_df,
                                   embeddings, input_type='similarity',
                                   metric='Acc'):
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

        if metric == 'Acc':
            y_pred = model.predict(X)
            values[concept] = accuracy_score(y, y_pred)
        elif metric == 'F1':
            y_pred = model.predict(X)
            values[concept] = f1_score(y, y_pred)
        elif metric == 'AUC':
            y_score = model.predict_proba(X)[:,1]
            values[concept] = roc_auc_score(y, y_score)
        elif 'K' in metric:
            y_score = model.predict_proba(X)[:,1]
            values[concept] = calibration_error(y, y_score, measure=metric)
    return values

def get_all_models_classification_metric(base_models, metadata_df, cosine_similarity_df,
                                         embeddings, metric='Acc', input_type='similarity'):
    model_names = base_models.keys()
    values_list = []
    for model_name in model_names:
        if 'Emb' in model_name:
            values = get_test_classification_metric(base_models[model_name], metadata_df, cosine_similarity_df,
                           embeddings, input_type='embeddings', metric=metric)
        elif ('GT' in model_name or 'CT' in model_name) and (metric=='AUC' or 'K' in metric):
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
                                   metric='Acc'):
    base_models_df = get_all_models_classification_metric(base_models, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric)
    base_models_df = base_models_df.reset_index()
    base_models_df['Calibration'] = 'None'
    
    cal_m3_models_df = get_all_models_classification_metric(m3_models_cal, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric)
    cal_m3_models_df = cal_m3_models_df.reset_index()
    cal_m3_models_df['Calibration'] = cal_m3_models_df['Model']
    cal_m3_models_df['Model'] = 'GLR'
    
    cal_m4_models_df = get_all_models_classification_metric(m4_models_cal, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric)
    cal_m4_models_df = cal_m4_models_df.reset_index()
    cal_m4_models_df['Calibration'] = cal_m4_models_df['Model']
    cal_m4_models_df['Model'] = 'CLR'
    
    cal_m5_models_df = get_all_models_classification_metric(m5_models_cal, test_metadata_df, test_cosine_similarity_df,
                                                             test_embeddings, metric=metric, input_type='embeddings')
    cal_m5_models_df = cal_m5_models_df.reset_index()
    cal_m5_models_df['Calibration'] = cal_m5_models_df['Model']
    cal_m5_models_df['Model'] = 'EmbCLR'

    df = pd.concat([base_models_df, cal_m3_models_df, cal_m4_models_df, cal_m5_models_df])
    df['Calibration'] = pd.Categorical(df['Calibration'], calibration_ordering)
    df = df.sort_values(by=['Model', 'Calibration'])
    df = df.set_index(['Model','Calibration'])
    df = df.loc[models_order]
    return df

def compare_all_models_calibration_avg(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings):
    metrics_df = pd.DataFrame()
    for metric in ['Acc', 'F1', 'AUC', 'K1', 'K2', 'Kmax']:
        df = compare_all_models_calibration_metric(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings, 
                                   metric=metric)
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace('-', np.nan).infer_objects(copy=False)
        series_mean = df.transpose().mean(skipna=True)
        series_std = df.transpose().std(skipna=True)
        series_str = series_mean.apply(lambda x: '{0:.3f}'.format(x))+u" \u00B1 "+series_std.apply(lambda x: '{0:.3f}'.format(x))
        metrics_df[metric] = series_str
    metrics_df = metrics_df.replace(u"nan \u00B1 nan", '-')
    metrics_df = metrics_df.loc[models_order]
    return metrics_df

def compare_all_models_calibration_concept(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                           test_metadata_df, test_cosine_similarity_df, test_embeddings,
                                          concept):
    metrics_df = pd.DataFrame()
    for metric in ['Acc', 'F1', 'AUC', 'K1', 'K2', 'Kmax']:
        df = compare_all_models_calibration_metric(base_models, m3_models_cal, m4_models_cal, m5_models_cal,
                                   test_metadata_df, test_cosine_similarity_df, test_embeddings, 
                                   metric=metric)
        with pd.option_context("future.no_silent_downcasting", True):
            df = df.replace('-', np.nan).infer_objects(copy=False)
        series = df[concept]
        metrics_df[metric] = series.apply(lambda x: '{0:.3f}'.format(x))
    metrics_df = metrics_df.replace('nan', '-')
    metrics_df = metrics_df.loc[models_order]
    return metrics_df

def plot_metrics_by_concept(models_dict, test_metadata_df, test_cosine_similarity_df, 
                            test_embeddings, save_path, dataset_name, concepts=None):
    if concepts is None:
        concepts = list(test_cosine_similarity_df.columns)
    select_models = list(models_dict.keys())
    palette = {select_models[0]: '#3182bd', 
               select_models[1]: '#9ecae1',
               select_models[2]: '#e6550d',
               select_models[3]: '#fdae6b'
              }
    
    metric_map = {'Acc': 'Accuracy', 'K1': 'Mean calibration error', 
                  'Kmax': 'Maximum calibration error'}
    
    fig, axs = plt.subplots(1,3, figsize=(9,4), sharey=True)
    
    for i, metric in enumerate(['Acc','K1','Kmax']):
        metrics_d = {'Model': [],
                     'Concept': [],
                      metric_map[metric]: []
                    }
        for m_name, m in models_dict.items():
            if 'Emb' in m_name:
                metrics_dict = get_test_classification_metric(m, test_metadata_df, 
                                                              test_cosine_similarity_df, 
                                                              test_embeddings, 
                                                               metric=metric,
                                                             input_type='embeddings')
            else:
                metrics_dict = get_test_classification_metric(m, test_metadata_df, 
                                                              test_cosine_similarity_df, 
                                                              test_embeddings, 
                                                               metric=metric)
            for concept in concepts:
                metrics_d['Model'].append(m_name)
                metrics_d['Concept'].append(concept)
                metrics_d[metric_map[metric]].append(metrics_dict[concept])
                
        metric_df_aux = pd.DataFrame(metrics_d)
        
        sns.barplot(metric_df_aux, 
                    y="Concept", x=metric_map[metric], hue="Model", 
                    legend=True, palette=palette, 
                   ax=axs[i])
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].get_legend().remove()
    lgd = fig.legend(handles, labels, ncols=4, loc = "upper center", bbox_to_anchor = (0.5, 0),
                    title='Model')
    tl = fig.suptitle(f'Breakdown of metrics by concept - {dataset_name}')
    fig.tight_layout()
    fig.savefig(save_path+f'metrics_concept.png', bbox_extra_artists=(tl,lgd),
                    bbox_inches='tight', dpi=150)
    return fig