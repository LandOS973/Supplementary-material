import numpy as np


class Abstract_EDA:

    def __init__(self, N, lambda_, device):
        self.N = N
        self.lambda_ = lambda_
        self.device = device


    def sample_solutions(self):
        pass

    def updateDistribution(self, solutionList, scoreList):
        pass

    def reset_learned_parameters(self, nb_instances):
        pass

    def get_init_hyperparameters(self):
        pass

    def get_lower_bound_hyperparameters(self):
        pass

    def get_upper_bound_hyperparameters(self):
        pass


    def update_hyperparameters(self, params):
        pass


    def rescale_hyperpameters(self, params):

        a = self.get_lower_bound_hyperparameters()
        b = self.get_upper_bound_hyperparameters()

        
        
        init_solution_cmaes = 1 / np.pi * np.arccos(1 - 2 * (params - a) / (b - a))

        return init_solution_cmaes


    def unrescale_hyperpameters(self, params):

        a = self.get_lower_bound_hyperparameters()
        b = self.get_upper_bound_hyperparameters()

        hyperparameters_config = a + (b - a) * (1 - np.cos(np.pi * params )) / 2

        return hyperparameters_config






