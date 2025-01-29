'''
2. Concept Model Utils
    c) Calibrators
'''
from .concept_bottleneck_models import ConceptModel

class CalibratedConceptModel(ConceptModel):
    def __init__(self, base_model, method, individual, info = {}):
        """
        base model (ConceptModel) : the uncalibrated concept classifier
        method (str)              : name of the calibration method
        individual (bool)         : whether to calibrate each concept separately (True) or not (False)
        info (dict)               : whatever notes you wanna put in there
        """
        super(CalibratedConceptModel, self).__init__(base_model.concepts)
        self.base_model = base_model
        self.method = method # string
        self.individual = individual
        self.info = info

    def _make_calibrator(self):
        if self.method == 'Histogram':
            self.logits = False
            self.calibrator_type = 'netcal'
            from netcal.binning import HistogramBinning as HB
            nbins = self.info['nbins'] if 'nbins' in self.info.keys() else 10
            return HB(nbins, detection = False)
        elif self.method == 'Isotonic':
            self.logits = False
            self.calibrator_type = 'netcal'
            from netcal.binning import IsotonicRegression as IR
            return IR(detection = False)
        elif self.method == 'Temperature':
            self.logits = False # I guess it does this itself?
            self.calibrator_type = 'netcal'
            from netcal.scaling import LogisticCalibration as LC
            return LC(temperature_only = True, method = "mle", detection = False)
        elif self.method == 'Platt':
            self.logits = False # I guess it does this itself?
            self.calibrator_type = 'netcal'
            from netcal.scaling import LogisticCalibration as LC
            return LC(temperature_only = False, method = "mle", detection = False)
        elif self.method == 'Beta':
            self.logits = False
            self.calibrator_type = 'netcal'
            from netcal.scaling import BetaCalibration as BC
            return BC(method = "mle", detection = False)
        
        # regularized calibration methods
        elif self.method.startswith('Platt') or self.method.startswith('Temperature'):
            self.logits = True
            self.calibrator_type = 'sklearn'
            method, pen = self.method.split(' ')
            method, pen = method.lower(), pen.lower()
            
            fit_intercept = method == 'platt'
            solver_map = {None : 'lbfgs', 'l2' : 'lbfgs', 'l1' : 'liblinear', 'elasticnet' : 'saga'}
            solver = solver_map[pen]
            
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(penalty = pen, solver = solver, fit_intercept = fit_intercept, max_iter = 200)
        else:
            raise NotImplementedError()
            
    def _make_calibrators(self):
        if self.individual:
            self.calibrators = {}
            for concept in self.concepts:
                self.calibrators[concept] = self._make_calibrator()
        else:
            self.calibrator = self._make_calibrator()

    def _get_math_params(self, calibrator):
        params = {}
        if self.method == 'Histogram':
            params['bin_bounds'] = calibrator.get_params()['_bin_bounds'][0]
            params['bin_vals'] = calibrator.get_params()['_bin_map']
        elif self.method == 'Isotonic':
            # idk bro
            params = {}
        elif self.method == 'Temperature':
            params['T'] = 1 / calibrator.weights[0]
        elif self.method == 'Platt':
            params['A'] = calibrator.weights[0]
            params['B'] = calibrator.intercept[0]
        elif self.method == 'Beta':
            tmp = calibrator.get_params()['_sites']
            a, b = tmp['weights']['values']
            c = tmp['bias']['values'][0]
            params = {'a' : a, 'b' : b, 'c' : c}
        elif self.method.startswith('Platt'):
            params['A'] = calibrator.coef_[0][0]
            params['B'] = calibrator.intercept_[0]
        elif self.method.startswith('Temperature'):
            params['T'] = 1 / calibrator.coef_[0][0]
        else:
            raise NotImplementedError()
        return params

    def preprocess_data(self, X):
        if self.calibrator_type == 'netcal':
            return X
        elif self.calibrator_type == 'sklearn':
            return X.reshape(-1, 1)
        else:
            raise ValueError('Unrecognized calibrator type: {}'.format(self.calibrator_type))

    def calibrate(self, X, y):
        self._make_calibrators()
        
        if self.logits:
            z = self.base_model.decision_function(X) # (n_cal, k,)
        else:
            z = self.base_model.predict_proba(X)

        self.params = {}
        
        if self.individual:
            for i, concept in enumerate(self.concepts):
                self.calibrators[concept].fit(self.preprocess_data(z[:, i]), y[:, i])
                self.params[concept] = self._get_math_params(self.calibrators[concept])
        else:
            self.calibrator.fit(self.preprocess_data(z.flatten()), y.flatten())
            self.params = self._get_math_params(self.calibrator)

    def transform(self, calibrator, X):
        if self.calibrator_type == 'netcal':
            return calibrator.transform(self.preprocess_data(X))
        elif self.calibrator_type == 'sklearn':
            return calibrator.predict_proba(self.preprocess_data(X))[:, 1]
        else:
            raise ValueError('Unrecognized calibrator type: {}'.format(self.calibrator_type))
    
    def predict_proba(self, X):
        z = self.base_model.predict_proba(X)
        if self.logits:
            z = self.base_model.decision_function(X)
        else:
            z = self.base_model.predict_proba(X)

        if self.individual:
            for i, concept in enumerate(self.concepts):
                z[:, i] = self.transform(self.calibrators[concept], z[:, i])
        else:
            #z = self.calibrator.transform(z) # shape issue
            for i, concept in enumerate(self.concepts):
                z[:, i] = self.transform(self.calibrator, z[:, i])
        
        return z
    

