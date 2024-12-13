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

def get_global_sim_X_y(metadata_df, cosine_similarity_df):
    X_list = []
    y_list = []
    concepts = list(cosine_similarity_df.columns)
    for concept in concepts:
        X_concept, y_concept = get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept)
        X_list.append(X_concept)
        y_list.append(y_concept)
    # Concatenate train data
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list)
    return X, y, X_list, y_list

# --------------------------
# Threshold-based methods
# --------------------------
class ThresholdModel:
    """
    Class for the threshold-based models. 
    Given a single input feature (the cosine similarity between the embeddings 
    with the concept vector), the threshold model has a single threshold paremeter
    to classify concept presence.
    The threshold is optimized for minimizing classification error (ie, 1 - accuracy).
    """ 
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
    """
    === M1 ===
    Trains ThresholdModel with a global threshold on cosine similarity to assign concept 
    presence for all concepts. Evaluates train error on all samples and individually on
    each concept.

    Parameters
    ----------
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        cosine_similarity_df (pd.DataFrame): Dataframe with the cosine similarity between 
            train samples and concept vectors.
        verbose (bool, default=True): Option to print threshold and train errors.

    Returns
    -------
        model (ThresholdModel): Trained global threshold model.
        global_train_error (float): Train error for all samples.
        train_errors (dict of float): Dictionary of train error for each concept.
    """
    concepts = list(cosine_similarity_df.columns)
    X_train, y_train, X_list, y_list = get_global_sim_X_y(metadata_df, cosine_similarity_df)
    
    # Train global threshold model
    model = ThresholdModel()
    thresh, global_train_error = model.fit(X_train, y_train)
    if verbose:
        print(f'Global threshold: {thresh:.3f} | Train error: {global_train_error:.3f}')
    
    train_errors = {}
    for i, concept in enumerate(concepts):
        y_train_pred = model.predict(X_list[i])
        train_error = 1 - accuracy_score(y_list[i], y_train_pred)
        train_errors[concept] = train_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Train error: {train_error:.3f}')
    return model, global_train_error, train_errors

def get_concept_threshold(metadata_df, cosine_similarity_df, concept):
    """
    Trains ThresholdModel for a single concept.

    Parameters
    ----------
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        cosine_similarity_df (pd.DataFrame): Dataframe with the cosine similarity between 
            train samples and concept vectors.
        concept (str): Concept name.

    Returns
    -------
        model (ThresholdModel): Trained threshold model for the concept.
        train_error (float): Train error for the concept samples.
    """
    X_train, y_train = get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept)
    model = ThresholdModel()
    thresh, train_error = model.fit(X_train, y_train)
    return model, train_error

def get_individual_thresholds(metadata_df, cosine_similarity_df, verbose=True):
    """
    === M2 ===
    Trains separate ThresholdModel for each of the concepts. Evaluates train error 
    individually on each concept.

    Parameters
    ----------
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        cosine_similarity_df (pd.DataFrame): Dataframe with the cosine similarity between 
            train samples and concept vectors.
        verbose (bool, default=True): Option to print train errors.

    Returns
    -------
        models (dict of ThresholdModel): Dictionary with the ThresholdModel model trained 
            for each concept.
        train_errors (dict of float): Dictionary of train error for each concept.
    """
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
    """
    Trains a LogisticRegression model given training data.
    Will be used for training the models based on cosine similarity and embeddings.

    Parameters
    ----------
        X_train (torch.tensor or np.array): Input features of the model.
        y_train (np.array): Binary array of concept presence.

    Returns
    -------
        model (sklearn.LogisticRegression): Trained logistic regression.
        train_error (float): Classification error on the training set.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    return model, train_error

def get_global_similarity_log_reg(metadata_df, cosine_similarity_df, verbose=True):
    """
    === M3 ===
    Trains a global LogisticRegression model using the cosine similarity as input feature.
    Evaluates train error on all samples and individually on each concept.

    Parameters
    ----------
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        cosine_similarity_df (pd.DataFrame): Dataframe with the cosine similarity between 
            train samples and concept vectors.
        verbose (bool, default=True): Option to print train errors.

    Returns
    -------
        model (sklearn.LogisticRegression): LogisticRegression model trained for all concepts
            with cosine similarity as feature.
        global_train_error (float): Train error for all samples.
        train_error (float): Classification error on the training set.
    """
    concepts = list(cosine_similarity_df.columns)
    X_train, y_train, X_list, y_list = get_global_sim_X_y(metadata_df, cosine_similarity_df)

    model, global_train_error = train_log_reg(X_train, y_train)
    if verbose:
        print(f'Global train error: {global_train_error:.3f}')
    
    train_errors = {}
    for i, concept in enumerate(concepts):
        y_train_pred = model.predict(X_list[i])
        train_error = 1 - accuracy_score(y_list[i], y_train_pred)
        train_errors[concept] = train_error
        if verbose:
            print(f'Concept: {concept.ljust(10)} | Train error: {train_error:.3f}')
    return model, global_train_error, train_errors

def get_concept_similarity_log_reg(metadata_df, cosine_similarity_df, concept):
    """
    Trains a LogisticRegression model for a single concept using the cosine similarity 
    as input feature.

    Parameters
    ----------
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        cosine_similarity_df (pd.DataFrame): Dataframe with the cosine similarity between 
            train samples and concept vectors.
        concept (str): Concept name.

    Returns
    -------
        model (sklearn.LogisticRegression): LogisticRegression model trained for the concept 
            with cosine similarity as feature.
        train_error (float): Classification error on the training set.
    """
    X, y = get_concept_sim_X_y(metadata_df, cosine_similarity_df, concept)
    model, train_error = train_log_reg(X, y)
    return model, train_error

def get_similarity_log_reg(metadata_df, cosine_similarity_df, verbose=True):
    """
    === M4 ===
    Trains separate LogisticRegression for each concept using the cosine similarity as 
    input feature. Evaluates train error individually on each concept.

    Parameters
    ----------
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        cosine_similarity_df (pd.DataFrame): Dataframe with the cosine similarity between 
            train samples and concept vectors.
        verbose (bool, default=True): Option to print train errors.

    Returns
    -------
        models (dict of sklearn.LogisticRegression): Dictionary with LogisticRegression 
            model trained for each concept with cosine similarity as feature.
        train_errors (dict of float): Dictionary of train error for each concept.
    """
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
    """
    Trains a LogisticRegression model for a single concept using the embeddings
    as input features.

    Parameters
    ----------
        embeddings (torch.Tensor): Tensor with the embedding of each train sample.
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        concept (str): Concept name.

    Returns
    -------
        model (sklearn.LogisticRegression): LogisticRegression model trained for the concept 
            with embeddings as features.
        train_error (float): Classification error on the training set.
    """
    X = embeddings
    y = (metadata_df[concept]==1).to_numpy().astype(int)
    model, train_error = train_log_reg(X, y)
    return model, train_error

def get_embeddings_log_reg(embeddings, metadata_df, cosine_similarity_df, verbose=True):
    """
    === M5 ===
    Trains separate LogisticRegression for each concept using the embeddings as 
    input features. Evaluates train error individually on each concept.

    Parameters
    ----------
        embeddings (torch.Tensor): Tensor with the embedding of each train sample.
        metadata_df (pd.DataFrame): Dataframe with train set metadata.
        verbose (bool, default=True): Option to print train errors.

    Returns
    -------
        models (dict of sklearn.LogisticRegression): Dictionary with the LogisticRegression 
            model trained for each concept with embeddings as featues.
        train_errors (dict of float): Dictionary of train error for each concept.
    """
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