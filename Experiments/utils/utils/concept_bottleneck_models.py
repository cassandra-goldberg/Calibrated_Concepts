'''
2. Concept Model Utils
    b) Models
'''
import numpy as np
from scipy.special import logit, expit
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# parent class
class ConceptModel(ClassifierMixin, BaseEstimator):
    """
    Base parent class for the X-to-concept layer of a concept bottleneck model. Should not be instantiated directly.
    """
    
    def __init__(self, concepts, seed = None, info = None):
        """
        Construct a ConceptModel instance (the X-to-concept layer of a concept bottleneck model)
        === Parameters ===
        concepts : array-like of string-type elements
            List of concept names
        model_constructor : str -> nn.Module, optional, default: MLP constructor
            Function mapping a concept name to a model instance, to be used to predict the presence of that concept on data samples
        seed : int, optional, default: None (random seeds for each concept model)
            Random seed to be used in model construction. If None, no seed is provided and the classifier uses a random seed
        info : Dict
            General metadata/info. Good to include name, description, etc.
        """
        super(ConceptModel, self).__init__()
        self.concepts = concepts
        self.seed = seed
        self.info = info
        self.params = {"concepts" : concepts, "seed" : seed, "info" : info}
        
        self._is_fitted = False
    
    def fit(self, X, y, verbose = False):
        """
        Fit all concept models to training dataset
        === Parameters ===
        X : ndarray-like (n, d) where n is the number of samples and d is the dimension of elements in the input space)
            Array of input features of whatever type is acceptable to the individual concept models (probably float arrays)
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Array of 1-0 concept labels for X
        verbose : bool, optional, default: False
            Whether to print progress during training
        """
        raise NotImplementedError()
    
    def predict_proba(self, X):
        """
        Predict concept probabilities on a dataset
        === Parameters ===
        X : ndarray-like (n, d) where n is the number of samples and d is the dimension of elements in the input space)
            Array of input features of whatever type is acceptable to the individual concept models (probably float arrays)
        === Output ===
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Array of concept probabilities in [0, 1] for X
        """
        raise NotImplementedError()
    
    def predict_log_proba(self, X, eps = 0.000001):
        """
        Predict log concept probabilities on a dataset
        === Parameters ===
        X : ndarray-like (n, d) where n is the number of samples and d is the dimension of elements in the input space)
            Array of input features of whatever type is acceptable to the individual concept models (probably float arrays)
        === Output ===
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Array of concept probabilities in [0, 1] for X
        """
        return np.log(np.clip(self.predict_proba(X), eps, 1 - eps))

    def decision_function(self, X, eps = 0.000001):
        """
        Compute non-probabilistic concept scores on a dataset
        === Parameters ===
        X : ndarray-like (n, d) where n is the number of samples and d is the dimension of elements in the input space)
            Array of input features of whatever type is acceptable to the individual concept models (probably float arrays)
        === Output ===
        z : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Array of non-probabilistic concept scores on the real line for X
        """
        return logit(np.clip(self.predict_proba(X), eps, 1 - eps))

    def predict(self, X):
        """
        Predict concept labels on a dataset, given similarity scores
        === Parameters ===
        X : ndarray-like (n, d) where n is the number of samples and d is the dimension of elements in the input space)
            Array of input features of whatever type is acceptable to the individual concept models (probably float arrays)
        === Output ===
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Matrix of 0-1 concept indicators for X
        """
        return self.predict_proba(X) > 0.5

    def score(self, X, y):
        """
        Predict concept labels on a dataset, given similarity scores
        === Parameters ===
        X : ndarray-like (n, d) where n is the number of samples and d is the dimension of elements in the input space)
            Array of input features of whatever type is acceptable to the individual concept models (probably float arrays)
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Matrix of 0-1 concept indicators for X
        === Output ===
        Accuracy score (float in range [0, 1])
        """
        return (self.predict(X) == y).mean()

    def __sklearn_is_fitted__(self):
        return self._is_fitted
    
    def get_params(self, deep=True):
        return self.params
    
    def set_params(self, params):
        for key in params.keys():
            self.params[key] = params[key]
            setattr(self.params, key, params[key])
        self._is_fitted = False

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.two_d_labels    = True
        tags.target_tags.multi_output    = True
        tags.target_tags.single_output   = False
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.multi_label = True
        
        return tags


# used for M1
class CosineSimilarityModel(ConceptModel):
    """
    Preprocessed-X-to-concept layer of a concept bottleneck model. For raw input domain X and k concepts, the
    input type for this module is a (k,)-dimensional 1D-vector of scalar similarity scores on the real line, 
    e.g., cosine similarities to concept activation vectors.

    Used for M1 (Logistic Cosine Similarity first-stage model).
    """
    
    def __init__(self, concepts, seed = None, info = None):
        super(CosineSimilarityModel, self).__init__(concepts, seed, info)
        
        self.concept_models = {}
        for concept in self.concepts:
            self.concept_models[concept] = LogisticRegression(random_state=seed, max_iter=1000)

    def fit(self, X, y, verbose = False):
        """
        Fit all concept models to training dataset (given X = train_cosine_similarity_df and y = train_concepts)
        === Parameters ===
        X : dataframe of shape (n, k) where n is the number of samples and k is the number of concepts in the concept space
            Matrix of similarity scores
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Matrix of 1-0 concept labels for X
        verbose : bool, optional, default: False
            Whether to print progress during training
        """
        # 1. preprocess
        if type(X) is pd.core.frame.DataFrame:
            X = X.to_numpy()
        # 2. do the thing
        for i, concept in enumerate(self.concepts):
            X_concept = X[:, i].reshape(-1, 1) # X[concept].to_numpy().reshape(-1, 1)
            y_concept = y[:, i]
            if verbose:
                print("{:03d}. Training {} detector with {} positive examples (of {})".format(i, concept, y_concept.sum(), len(y_concept)))
            self.concept_models[concept].fit(X_concept, y_concept)

        # 3. bookkeeping
        self._is_fitted = True
        self.classes_ = self.concepts
        self.X_ = X
        self.y_ = y

    def predict_proba(self, X, concept = None):
        """
        Predict concept probabilities on a dataset, given similarity scores
        === Parameters ===
        X : dataframe of shape (n, k) where n is the number of samples and k is the number of concepts in the concept space
            Matrix of similarity scores
        concept : string, optional, default: None
            If not None, name of the concept to return probabilities for (acting as a probabilistic single-output binary classifier)
        === Output ===
        y : ndarray-like (n, k) where n is the number of samples and k is the number of concepts in the concept space)
            Matrix of concept probabilities in [0, 1] for X
        """
        # preprocess
        if type(X) is pd.core.frame.DataFrame:
            X = X.to_numpy()
        
        # predict all concepts
        if concept is None:
            y = np.empty(shape=(len(X), len(self.concepts)))
            for i, concept in enumerate(self.concepts):
                y[:, i] = self.concept_models[concept].predict_proba(X[:, i].reshape(-1, 1))[:, 1] # only need positive label probabilities
        
        # predict for a single concept
        else:
            i = self.concepts.index(concept)
            y = self.concept_models[concept].predict_proba(X[:, i].reshape(-1, 1))[:, 1] # only need positive label probabilities
        
        return y
        
    def __sklearn_clone__(self):
        r = CosineSimilarityModel(self.concepts.copy(), self.seed, self.info.copy())
        for concept in self.concepts:
            r.concept_models[concept] = self.concept_models[concept].__sklearn_clone__()
        r._is_fitted = True
        r.classes_ = r.concepts
        r.X_ = self.X_
        r.y_ = self.y_
        return r
    

# hypothetically used for M2, untested
class MLPConceptModel(ConceptModel):
    """
    X-to-concept layer of a concept bottleneck model. Untested, used for M2, very slow in practice.
    """
    def __init__(self, concepts, seed = None, info = None):
        super(MLPConceptModel, self).__init__(concepts, seed, info)
        
        self.model_constructor = lambda concept : MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200) if self.seed is None else MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=200, random_state = self.seed)
        
        self.concept_models = {}
        for concept in self.concepts:
            self.concept_models[concept] = self.model_constructor(concept)
    
    def fit(self, X, y, verbose = False):
        for i, concept in enumerate(self.concepts):
            y_concept = y[:, i]
            if verbose:
                print("{:03d}. Training {} detector with {} positive examples (of {})".format(
                    i, concept, y_concept.sum(), len(y_concept)))
            self.concept_models[concept].fit(X, y_concept)
        self._is_fitted = True

    def predict_proba(self, X):
        y = np.empty(shape=(len(X), len(self.concepts)))
        for i, concept in enumerate(self.concepts):
            y[:, i] = self.concept_models[concept].predict_proba(X)[:, 1] # only need positive label probabilities
        return y


# used for M3
class RandomForestConceptModel(ConceptModel):
    """
    X-to-concept layer of a concept bottleneck model. For raw input domain X and k concepts, the
    input type for this module is a 1D-vector embedding, e.g., obtained from a foundation model.

    Used for M3 (Embeddings Random Forest first-stage model), still kinda slow.
    """
    def __init__(self, concepts, n_estimators = 100, max_depth = 3, seed = None, info = None):
        super(RandomForestConceptModel, self).__init__(concepts, seed, info)
        self.max_depth = max_depth
        self.n_estimators = n_estimators

        if seed is None:
            self.model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
        else:
            self.model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = seed)
    
    def fit(self, X, y, verbose = False):
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X):
        return np.array(self.model.predict_proba(X))[:, :, 1].T
    