
import torch
from eda_strategies.Abstract_EDA import Abstract_EDA

import numpy as np
from torch.distributions import kl_divergence

from utils.ppo_eda_utils import PPO_EDA_generator, OrderGenerator, LearnedOrderGenerator, MatrixSampler

from torch.distributions import Categorical

import torch.nn.utils.prune as prune
import copy

class PPO_EDA(Abstract_EDA):

    def __init__(self, N,  lambda_, beta,  device, typeModel, numberHiddenLayersG, nh, isUnivariate, dropoutGen, dropoutTrain, withoutCausalMaskTraining, dim_variables, learnDAG, noise_rescale):

        Abstract_EDA.__init__(self, N, lambda_, device)

        #self.typeOrder = typeOrder

        self.typeModel = typeModel

        self.isUnivariate = isUnivariate
        
        self.dim_variables = dim_variables

    
        
        if dim_variables is not None:
            self.max_dim = max(dim_variables)

        self.cpt = 0

        self.numberHiddenLayersG = numberHiddenLayersG

        self.nh = nh

        self.nb_train = 50
        self.lambda_ = lambda_
        
        self.epsilon = 0.001
        self.first_threshold = torch.nn.Threshold(self.epsilon, self.epsilon)
        self.second_threshold = torch.nn.Threshold(- 1 + self.epsilon, - 1 + self.epsilon)
        
        self.weights = torch.tensor(np.linspace(-1, 1, num=self.lambda_)).to(self.device)

        self.knownDAG = None
        self.knownOrder = None
        
        
        self.sameDagTraining = False
        
        self.beta = beta
        self.dropoutGen = dropoutGen
        self.dropoutTrain = dropoutTrain
        
        self.learnDAG = learnDAG
        
        self.noise_rescale = noise_rescale
        
        self.withoutCausalMaskTraining = withoutCausalMaskTraining
        


    def setKnownDAG(self, knownDAG):
        
        self.knownDAG = knownDAG
        
    

    def setKnownOrder(self, knownOrder):
        
        self.knownOrder = knownOrder
    

    def setSameDagTraining(self):
        
        self.sameDagTraining = True
        
            

    def reset_learned_parameters(self, nb_instances):

  
        self.nb_instances = nb_instances

        if(self.typeModel == "Linear"):
            self.generator = PPO_EDA_generator((self.nb_instances, self.lambda_, self.N), -1, self.lambda_,cat_sizes= self.dim_variables,  linear=True).to(self.device)
        elif(self.typeModel == "NeuralNet"):
            self.generator = PPO_EDA_generator((self.nb_instances, self.lambda_, self.N), self.nh,self.lambda_,skeleton=None,cat_sizes= self.dim_variables,linear=False, numberHiddenLayersG=self.numberHiddenLayersG, device=self.device).to(self.device)

        self.generator.reset_parameters()

        
        if(self.learnDAG):
            self.orderGenerator = LearnedOrderGenerator(self.device,  nb_instances, self.lambda_, self.N, self.noise_rescale).to(self.device)
            self.orderGenerator.reset_parameters()

        else:
        
            self.orderGenerator = OrderGenerator(self.device,  nb_instances, self.lambda_, self.N).to(self.device)

            
            if(self.knownDAG != None):
                self.orderGenerator.setKnownDAG(self.knownDAG)
                
            if(self.knownOrder != None):
                self.orderGenerator.setKnownOrder(self.knownOrder)
                    

        if(self.dropoutGen != 0.0):
            self.tensor_proba_mask_gen = torch.tensor(np.ones((self.nb_instances, self.lambda_, self.N, self.N))* (1-self.dropoutGen)).to(self.device).float()
        
        if(self.dropoutTrain != 0.0):
            self.tensor_proba_mask_train = torch.tensor(np.ones((self.nb_instances, self.lambda_, self.N, self.N))* (1-self.dropoutTrain)).to(self.device).float()
            
        if(self.withoutCausalMaskTraining):
            self.fullMask = 1 - torch.eye(self.N,self.N)
            self.fullMask = self.fullMask.unsqueeze(0).unsqueeze(1).repeat([nb_instances, self.lambda_, 1, 1]).to(self.device)       

        self.different_number_of_categories = False
        self.mask_categorical = False
        
        
        if (self.dim_variables is not None):
            
            if(len(set(self.dim_variables))!=1):
                
                self.different_number_of_categories = True

                self.mask_categorical = torch.zeros(self.N, self.max_dim)
                self.mask_categorical2 = torch.ones(self.N, self.max_dim)

                for idx, dim in enumerate(self.dim_variables):
                    self.mask_categorical[idx, dim:] = -float("inf")
                    self.mask_categorical2[idx, dim:] = 0

                self.mask_categorical = self.mask_categorical.unsqueeze(0).unsqueeze(0).repeat([nb_instances, self.lambda_, 1, 1]).to(self.device)
                self.mask_categorical2 = self.mask_categorical2.unsqueeze(0).unsqueeze(0).repeat([nb_instances, self.lambda_, 1, 1]).to(self.device)

            

    



    def sample_solutions(self):



        with torch.no_grad():


            new_pop = torch.zeros((self.nb_instances, self.lambda_, self.N)).to(self.device)

            
            order_variables, dag = self.orderGenerator.get_order(False)



            order_variables = order_variables.long().data
            


            self.DAG = dag.long().data


            
            if(self.isUnivariate == 1):
                self.DAG.zero_()
            
            
            if(self.dropoutGen != 0.0):
                
                B = torch.bernoulli(self.tensor_proba_mask_gen)
                self.mask_gen =  self.DAG.data*B
            else:
                self.mask_gen = self.DAG
                

            

            # Génération des valeurs des variables des solutions les unes après les autres (de 0 à n-1)
            for i in range(0, self.N):


                DAG_input = self.mask_gen.gather(3, order_variables[:, :, i].unsqueeze(2).data.unsqueeze(3).repeat(1, 1, self.N, 1))


                if (self.different_number_of_categories):
                    mask_output = self.mask_categorical.gather(2, order_variables[:, :, i].unsqueeze(2).data.unsqueeze(3).repeat(1, 1, 1, self.max_dim)).squeeze(2)
                    mask_output2 = self.mask_categorical2.gather(2, order_variables[:, :, i].unsqueeze(2).data.unsqueeze(3).repeat(1, 1, 1, self.max_dim)).squeeze(2)
                else:
                    mask_output = None

               
                probas = self.generator(new_pop, DAG_input.squeeze(),  mask_output, order_variables[:, :, i])

                probas = self.first_threshold(probas)
                probas = - self.second_threshold(-probas)

                if (self.different_number_of_categories): 
                    probas = probas*mask_output2

                


                if(self.dim_variables is not None):

                    categorical_dist = Categorical(probas)
                    variable_ouput = categorical_dist.sample().float()

                else:
                    variable_ouput = torch.bernoulli(probas)



                
                new_pop.scatter_(2, order_variables[:, :, i].unsqueeze(2), variable_ouput.unsqueeze(2))



        return new_pop.unsqueeze(3).data
    


    def toString(self):

        return "Strategy_PPO_EDA_"




    def updateDistribution(self, solutionList, scoreList):

        

        sorted, indices = torch.sort(scoreList, dim=1)


        input_pop = (solutionList.squeeze(3)).gather(1, indices.unsqueeze(2).repeat([1, 1, self.N])).detach()
        target = input_pop.data
        
        sorted_dag = self.DAG.gather(1, indices.unsqueeze(2).unsqueeze(3).repeat([1, 1, self.N, self.N])) 
        
        sorted_mask_gen = self.mask_gen.gather(1, indices.unsqueeze(2).unsqueeze(3).repeat([1, 1, self.N, self.N])) 





        sade_optimizer = torch.optim.Adam(list(self.generator.parameters()), lr=0.001)

   
        with torch.no_grad():
     

        
            init_distributions = self.generator(input_pop.data, sorted_mask_gen, self.mask_categorical).data
            init_distributions = self.first_threshold(init_distributions)
            init_distributions = - self.second_threshold(-init_distributions)
            
            old_distrib = init_distributions.data.clone().detach()
                
            if (self.dim_variables is not None):

                proba_action_init = init_distributions.gather(3, target.unsqueeze(3).long()).squeeze(3)

            else:
                proba_action_init = torch.where(target == 1, init_distributions, 1 - init_distributions)


        pbar = range(self.nb_train)



        if(self.learnDAG):
            orderGenerator_optimizer = torch.optim.Adam(list(self.orderGenerator.parameters()), lr=0.01)


        for epoch in pbar:

            sade_optimizer.zero_grad()
            
            if(self.learnDAG):
                orderGenerator_optimizer.zero_grad()

            if(self.sameDagTraining):
                
                dag = sorted_dag.data
                
            else:
                order_variables, drawn_mask = self.orderGenerator.get_order(True)
                dag = drawn_mask
            
            if(self.withoutCausalMaskTraining):
                dag = self.fullMask.data

            if(self.dropoutTrain != 0.0):
                
                B = torch.bernoulli(self.tensor_proba_mask_train)
                dag = dag*B
            
                    

            
            if(self.isUnivariate == 1):
                dag.zero_()  
                
                
            probas_g = self.generator(input_pop.data, dag, self.mask_categorical)
            probas_g = self.first_threshold(probas_g)
            probas_g = - self.second_threshold(-probas_g)

            generated_probas = probas_g


            if (self.dim_variables is not None):
                generated_probas_action = generated_probas.gather(3, target.unsqueeze(3).long()).squeeze(3)
            else:
                generated_probas_action = torch.where(target == 1, generated_probas,
                                                    1 - generated_probas)


            ratio = -generated_probas_action / proba_action_init.data

            weighted_loss = torch.transpose(ratio, 1, 2) * self.weights


            d_kl = kl_divergence(torch.distributions.bernoulli.Bernoulli(probs=init_distributions.data),
                                         torch.distributions.bernoulli.Bernoulli(probs=generated_probas)).mean()


            global_loss = torch.mean(weighted_loss)  + self.beta * d_kl

            global_loss.backward()

            sade_optimizer.step()
            
            
            if(self.learnDAG):

                orderGenerator_optimizer.step()
         

