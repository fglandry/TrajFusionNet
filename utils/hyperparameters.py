import copy
import itertools

HYPERPARAMS = {
    "trajectorytransformerb": {
        "VanillaTransformerForForecastClassification": {
            "n_heads": [12],
            "e_layers": [2, 4, 6, 8],
            "d_ff": [512, 1024],
            "d_model": [128, 256],
            "lr": [5.0e-04, 5.0e-05, 5.0e-06, 5.0e-07],
            "batch_size": [64, 32, 16, 8],
            "epochs": [30, 60, 90]
        }
    },
    "trajectorytransformer": {
        "VanillaTransformerForForecast": {
            "lr": [5.0e-05, 5.0e-06, 5.0e-07],
            "batch_size": [32, 16, 8]
            # "n_heads": [4, 8, 12],
            # "d_ff": [512, 1024],
            # "e_layers": [4, 8, 12],
            # "d_layers": [2, 4, 8, 12],
        }
    }
}

class HyperparamsOrchestrator():

    def __init__(self, tune_hyperparameters, model, submodel):
        global HYPERPARAMS
        self.current_case_idx = 0
        self.nb_cases = 1
        self.model = model
        self.submodel = submodel
        self.tune_hyperparameters = tune_hyperparameters
        if self.tune_hyperparameters:
            s = []
            params = HYPERPARAMS[model][submodel]
            for key, value in params.items():
                self.nb_cases = self.nb_cases * len(value)
                s.append(value)
            self.params_list = list(itertools.product(*s))
    
    def get_next_case(self):
        if not self.tune_hyperparameters:
            return {}
        case = copy.deepcopy(HYPERPARAMS)
        for idx, (key, value) in enumerate(HYPERPARAMS[self.model][self.submodel].items()):
            case[self.model][self.submodel][key] = self.params_list[self.current_case_idx][idx]
        self.current_case_idx += 1
        return case
