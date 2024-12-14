import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from calibration import calibration_error

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

def get_test_classification_metric(models, metadata_df, cosine_similarity_df,
                                   embeddings, input_type='similarity',
                                   metric='accuracy'):
    concepts = list(cosine_similarity_df.columns)
    values = {}
    model = models
    for concept in concepts:
        if input_type == 'similarity':
            X_concept = cosine_similarity_df[[concept]].to_numpy()
        elif input_type == 'embeddings':
            X_concept = embeddings
        y = (metadata_df[concept]==1).to_numpy().astype(int)
        if type(models) == dict:
            model = models[concept]

        if metric == 'accuracy':
            y_pred = model.predict(X_concept)
            values[concept] = accuracy_score(y, y_pred)
        elif metric == 'f1':
            y_pred = model.predict(X_concept)
            values[concept] = f1_score(y, y_pred)
        elif metric == 'auc':
            y_score = model.predict_proba(X_concept)[:,1]
            values[concept] = roc_auc_score(y, y_score)
        elif 'K' in metric:
            y_score = model.predict_proba(X_concept)[:,1]
            values[concept] = calibration_error(y, y_score, measure=metric)
    return values

def get_all_models_classification_metric(models, metadata_df, cosine_similarity_df,
                                         embeddings, metric='accuracy'):
    model_names = models.keys()
    values_list = []
    for model_name in model_names:
        if 'M5' in model_name:
            values = get_test_classification_metric(models[model_name], metadata_df, cosine_similarity_df,
                           embeddings, input_type='embeddings', metric=metric)
        elif 'Threshold' in model_name and (metric=='auc' or 'K' in metric):
            concepts = list(cosine_similarity_df.columns)
            values = dict.fromkeys(concepts, '-')
        else:
            values = get_test_classification_metric(models[model_name], metadata_df, cosine_similarity_df,
                           embeddings, input_type='similarity', metric=metric)
        values['Model'] = model_name
        values_list.append(values)
    comparison_df = pd.DataFrame.from_dict(values_list)
    comparison_df = comparison_df.set_index('Model')
    return comparison_df