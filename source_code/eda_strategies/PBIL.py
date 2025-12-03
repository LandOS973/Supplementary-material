

import torch
import numpy as np


from eda_strategies.UMDA import UMDA


class PBIL(UMDA):

    def __init__(self, N,  lambda_, device):

        UMDA.__init__(self,  N,  lambda_, device)


    def updateDistribution(self, solutionList, scoreList):

        solutions = (solutionList + 1) / 2

        sorted, indices = torch.sort(scoreList, dim=1)
        sorted_solutionList = (solutions.squeeze(3)).gather(1, indices.unsqueeze(2).repeat([1,1,self.N]))

        self.proba = (1-self.alpha) * self.proba + self.alpha *  torch.mean(sorted_solutionList[:, (self.lambda_ - self.mu):, :], dim=1).unsqueeze(1).unsqueeze(3)

        self.proba = self.first_threshold(self.proba)
        self.proba = - self.second_threshold(-self.proba)


    def toString(self):

        return "Strategy_PBIL"

    def get_lower_bound_hyperparameters(self):
        return np.array([1.0,0.0])

    def get_upper_bound_hyperparameters(self):
        return np.array([self.lambda_,1.0])

    def get_init_hyperparameters(self):

        a = self.get_lower_bound_hyperparameters()
        b = self.get_upper_bound_hyperparameters()

        return (a+ b)/2

    def update_hyperparameters(self, params):

        self.lambda_ = int(params[0])
        self.mu = int(params[1])
        self.alpha = params[2]

