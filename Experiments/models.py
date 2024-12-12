import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# --------------------------
# Utils
# --------------------------

def get_concept_X_y_list(metadata_df, cosine_similarity_df, concept):
    X = cosine_similarity_df[[concept]].to_numpy()
    y = (metadata_df[concept]==1).to_numpy().astype(int)
    concept_list = metadata_df[concept]
    return X, y, concept_list

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
    X_test_list = []
    y_test_list = []
    concepts = list(cosine_similarity_df.columns)
    # Split data for each concept
    for concept in concepts:
        X, y, concept_list = get_concept_X_y_list(metadata_df, cosine_similarity_df, concept)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, 
                                                            stratify=concept_list)
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
    # Concatenate train data
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list)
    # Train global threshold model
    model = ThresholdModel()
    thresh, global_train_error = model.fit(X_train, y_train)
    if verbose:
        print(f'Global threshold: {thresh:.3f} | Train error: {global_train_error:.3f}')
    # Evaluate on test data
    test_errors = {}
    for i, concept in enumerate(concepts):
        y_test_pred = model.predict(X_test_list[i], thresh)
        test_error = 1 - accuracy_score(y_test_list[i], y_test_pred)
        test_errors[concept] = test_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Test error: {test_error:.3f}')
    return thresh, global_train_error, test_errors

def get_concept_threshold(metadata_df, cosine_similarity_df, concept):
    X, y, concept_list = get_concept_X_y_list(metadata_df, cosine_similarity_df, concept)
    model = ThresholdModel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, 
                                                        random_state=42,
                                                        stratify=concept_list)
    thresh, train_error = model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test, thresh)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    return thresh, train_error, test_error

def get_individual_thresholds(metadata_df, cosine_similarity_df, verbose=True):
    concepts = list(cosine_similarity_df.columns)
    thresholds = {}
    train_errors = {}
    test_errors = {}
    for concept in concepts:
        thresh, train_error, test_error = get_concept_threshold(metadata_df, cosine_similarity_df, concept)
        thresholds[concept] = thresh
        train_errors[concept] = train_error
        test_errors[concept] = test_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Threshold: {thresh:.3f} | Train error: {train_error:.3f} | Test error: {test_error:.3f}')
    return thresholds, train_errors, test_errors

# --------------------------
# Linear methods
# --------------------------
def get_concept_similarity_log_reg(metadata_df, cosine_similarity_df, concept):
    X, y, concept_list = get_concept_X_y_list(metadata_df, cosine_similarity_df, concept)
    model = LogisticRegression(random_state=42, penalty=None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, 
                                                        random_state=42,
                                                        stratify=concept_list)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    return model, train_error, test_error

def get_similarity_log_reg(metadata_df, cosine_similarity_df, verbose=True):
    concepts = list(cosine_similarity_df.columns)
    train_errors = {}
    test_errors = {}
    for concept in concepts:
        _, train_error, test_error = get_concept_similarity_log_reg(metadata_df, cosine_similarity_df, concept)
        train_errors[concept] = train_error
        test_errors[concept] = test_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Train error: {train_error:.3f} | Test error: {test_error:.3f}')
    return train_errors, test_errors