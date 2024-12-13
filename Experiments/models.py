import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# --------------------------
# Utils
# --------------------------

def get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept):
    X = cosine_similarity_df[[concept]].to_numpy()
    y = (metadata_df[concept]==1).to_numpy().astype(int)
    return X, y

# --------------------------
# Threshold-based methods
# --------------------------
class ThresholdModel:
    def __init__(self):
        pass

    def predict(self, X_eval, thresh=None):
        if thresh is None:
            thresh = self.thresh
        return (X_eval[:,0] > thresh).astype(int)

    def loss(self, thresh):
        y_pred = self.predict(self.X, thresh)
        error = 1 - accuracy_score(self.y, y_pred)
        return error
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        result = minimize_scalar(self.loss, bounds=(-1, 1), method='bounded')
        self.thresh, self.loss = result.x, result.fun
        return self.thresh, self.loss

def get_global_threshold(metadata_df, cosine_similarity_df, verbose=True):
    X_train_list = []
    y_train_list = []
    concepts = list(cosine_similarity_df.columns)
    # Split data for each concept
    for concept in concepts:
        X_train, y_train = get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
    # Concatenate train data
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list)
    # Train global threshold model
    model = ThresholdModel()
    thresh, global_train_error = model.fit(X_train, y_train)
    if verbose:
        print(f'Global threshold: {thresh:.3f} | Train error: {global_train_error:.3f}')
    
    train_errors = {}
    for i, concept in enumerate(concepts):
        y_train_pred = model.predict(X_train_list[i])
        train_error = 1 - accuracy_score(y_train_list[i], y_train_pred)
        train_errors[concept] = train_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Train error: {train_error:.3f}')
    return model, global_train_error, train_errors

def get_concept_threshold(metadata_df, cosine_similarity_df, concept):
    X_train, y_train = get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept)
    model = ThresholdModel()
    thresh, train_error = model.fit(X_train, y_train)
    return model, train_error

def get_individual_thresholds(metadata_df, cosine_similarity_df, verbose=True):
    concepts = list(cosine_similarity_df.columns)
    models = {}
    train_errors = {}
    for concept in concepts:
        model, train_error = get_concept_threshold(metadata_df, cosine_similarity_df, concept)
        models[concept] = model
        train_errors[concept] = train_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Threshold: {model.thresh:.3f} | Train error: {train_error:.3f}')
    return models, train_errors

# --------------------------
# Linear methods
# --------------------------
def train_log_reg(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    return model, train_error

def get_concept_similarity_log_reg(metadata_df, cosine_similarity_df, concept):
    X, y = get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept)
    model, train_error = train_log_reg(X, y)
    return model, train_error

def get_similarity_log_reg(metadata_df, cosine_similarity_df, verbose=True):
    concepts = list(cosine_similarity_df.columns)
    models = {}
    train_errors = {}
    for concept in concepts:
        model, train_error = get_concept_similarity_log_reg(metadata_df, cosine_similarity_df, concept)
        train_errors[concept] = train_error
        models[concept] = model
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Train error: {train_error:.3f}')
    return models, train_errors

def get_concept_embeddings_log_reg(embeddings, metadata_df, concept):
    X = embeddings
    y = (metadata_df[concept]==1).to_numpy().astype(int)
    model, train_error = train_log_reg(X, y)
    return model, train_error

def get_embeddings_log_reg(embeddings, metadata_df, cosine_similarity_df, verbose=True):
    concepts = list(cosine_similarity_df.columns)
    models = {}
    train_errors = {}
    test_errors = {}
    for concept in concepts:
        model, train_error = get_concept_embeddings_log_reg(embeddings, metadata_df, concept)
        train_errors[concept] = train_error
        models[concept] = model
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Train error: {train_error:.3f}')
    return models, train_errors