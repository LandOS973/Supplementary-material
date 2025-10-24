import torch
import torch.nn as nn
from eda_strategies.Abstract_EDA import Abstract_EDA
import random


class UnivariatePPOEDA(Abstract_EDA, nn.Module):
    """
    Version univariée propre : chaque variable Xi est indépendante, 
    p(Xi,j=1) = sigmoid(theta_i_j) avec theta_i_j appris via une couche linéaire.
    i = instance, j = variable
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, learning_rate=0.01, device="cuda:0", agent_number=0):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.typeModel = typeModel
        self.learning_rate = learning_rate
        self.dim_variables = dim_variables
        self.lambda_ = lambda_
        self.N = N # nombre de variables
        self.device = device
        self.beta = beta
        self.theta = None
        self.optimizerG = None
        self.baseline = None
        self.agent_number = agent_number
        self.theta_old = None
    def forward(self):
        """
        Retourne les probabilités p(x_i=1) = sigmoid(theta_i)
        theta_i sont les poids de la couche linéaire.
        """
        probs = torch.sigmoid(self.theta)
        return probs  # (N,)

    def reset_learned_parameters(self, nb_instances):
        """Réinitialise les paramètres et sauvegarde le nombre d'instances."""
        self.theta = nn.Parameter(torch.zeros((nb_instances, self.N), device=self.device))
        self.theta_old = self.theta.detach().clone()
        self.nb_instances = nb_instances
        self.optimizerG = torch.optim.Adam([self.theta], lr=self.learning_rate)

    def sample_solutions(self):
        """
        Échantillonne λ solutions selon Bernoulli(p).
        Sortie : (nb_instances, λ, N, 1)
        """
        probs = self.forward() # (nb_instances, N)
        probs_pop = probs.unsqueeze(1).expand(-1, self.lambda_, -1)        # (nb_instances, λ, N)

        with torch.no_grad():
            samples = torch.bernoulli(probs_pop).unsqueeze(-1)             # (nb_instances, λ, N, 1)

        return samples.to(self.device)
    
    # Version Proximal Policy Optimization PPO with Clipped Objective
    # def updateDistribution(self, solutionList, scoreList):
    #     device = self.device
    #     scoreList = scoreList.to(device) 
    #     N = self.nb_instances
    #     K_steps = 4
    #     eps = 0.2
    #     total_loss = 0.0
    #     with torch.no_grad():
    #         theta_old = self.theta.detach().clone()
    #         pi_theta_old = torch.sigmoid(theta_old)  # (N, nb_vars)
    #     for n in range(N):
    #         Dk = solutionList[n].squeeze(-1)  # (λ, nb_vars)
    #         fitnesses = scoreList[n]  # (λ,)
    #         advantages = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-10) 
    #         for step in range(K_steps):
    #             pi_theta = torch.sigmoid(self.theta[n])
    #             log_pi_selected_old = torch.where(
    #                 Dk == 1.0,
    #                 torch.log(pi_theta_old[n] + 1e-10),
    #                 torch.log(1.0 - pi_theta_old[n] + 1e-10)
    #             )
    #             log_pi_selected = torch.where(
    #                 Dk == 1.0,
    #                 torch.log(pi_theta + 1e-10),
    #                 torch.log(1.0 - pi_theta + 1e-10)
    #             )

    #             log_pi_theta_xi = torch.sum(log_pi_selected, dim=1)
    #             log_pi_theta_old_xi = torch.sum(log_pi_selected_old, dim=1)
    #             ratio = torch.exp(log_pi_theta_xi - log_pi_theta_old_xi)

    #             clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
    #             loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))

    #             self.optimizerG.zero_grad()
    #             loss.backward()
    #             self.optimizerG.step()

    #         total_loss += loss.item()

    #     return total_loss / N

    # Version REINFORCE non optimisée
    # def updateDistribution(self, solutionList, scoreList):
    #     device = self.device
    #     solutionList = solutionList.to(device)
    #     scoreList = scoreList.to(device)
    #     solutions = solutionList.squeeze(-1)  # (nb_instances, λ, N)
    #     # X => Individu ayant N variables {-1,1} [x1, x2, ..., xN] => exemple dans QUBO 64, chaque individu est une solution de 64 variables
    #     X = solutions  # (nb_instances, λ, N)
    #     # i indice sur l'individu Xi [indi 1, indi 2, ..., indi λ]
    #     # k indice sur la variable de l'individu i [x1, x2, xK, xN] de l'individu Xi

    #     # L(Theta) = (1/self.lamba_) * sum( de i=1 a self.lambda_) [ fitness(Xi) * sum( de k=1 a N ) [ log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 ] ]
    #     # on fait les descentes de gradient des L(Theta_i) en parallèle pour chaque instance i a la fin des itérations
    #     total_loss = 0.0
    #     # probs = theta a qui on applique sigmoid pour avoir les proba de chaque variable
    #     probs = self.forward()  # (nb_instances, N)
    #     for n in range(self.nb_instances):
    #         #sum( de i=1 a self.lambda_)
    #         loss_n = 0.0
    #         # probabilités pour chaque variable de l'instance n
    #         # Theta => [Theta1, Theta2, ..., ThetaN] vecteur des probabilités pour chaque variable
    #         Pi_Theta = probs[n]  # (N,)
    #         # maintenant, on rentre dans chaque individu de l'instance n
    #         for i in range(self.lambda_): # i => indice sur l'individu {0,...,self.lambda_-1}
    #             # self.lambda_ => nombre d'individus samplés
    #             All_Pi_Theta_k_a_i_k = []
    #             # Log(Pi(Theta X)) = sum(log(Pi(Theta k)(a[k]))) = sum( log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 )
    #             log_Pi_Theta_X = 0.0
                
    #             # on rentre maintenant dans chaque variable de l'individu k de l'instance n
    #             for k in range(self.N): # k => indice sur la variable {0,...,N-1} (64 dans QUBO 64)
    #                 # a[i][k] => action choisie pour la variable k de l'individu i {0,1} => valeur tirée par la proba (convertie plus tard en {-1,1} pour le calcul de la fitness)
    #                 a_i_k = X[n, i, k]
    #                 # Pi(Theta k) => proba de choisir 1 pour la variable k pour chaque individu
    #                 Pi_Theta_k = Pi_Theta[k]
    #                 # Pi(Theta k)(a[i][K]) => proba de choisir l'action a[i][k] pour chaque individu equivaut ici a Sigmoid(theta k) si a[k]=1 et 1-Sigmoid(theta k) si a[k]=-1
    #                 # Pi_Theta = sigmoid(theta)
    #                 Pi_Theta_k_a_i_k = Pi_Theta_k if a_i_k.item() == 1.0 else (1.0 - Pi_Theta_k)
    #                 log_Pi_Theta_X += torch.log(Pi_Theta_k_a_i_k)
    #                 All_Pi_Theta_k_a_i_k.append(Pi_Theta_k_a_i_k)
                
    #             # Pi(Theta X) => proba de trouver l'individu X = Pi(Theta1)(a[1]) * Pi(Theta2)(a[2]) * ... * Pi(ThetaN)(a[N])
    #             # Pi_Theta_X = torch.prod(torch.stack(All_Pi_Theta_k_a_i_k))
    #             fitness_X_i = scoreList[n, i]
    #             # fitness(Xi) * sum( de k=1 a N ) [ log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 ] ]
    #             loss_n += fitness_X_i * log_Pi_Theta_X
    #         # J(Theta) => Esperance de la trajectoire générée par Pi(Theta X) sur la fitness
    #         # On veut maximiser J(Theta)
    #         # J(Theta) = Esperance~PiTheta [fitness(X)] => 1/self.lambda_ * sum( de i=1 a self.lambda_) [ fitness(Xi) ]
    #         total_loss += loss_n

    #     # L(Theta) = (1/self.lamba_) * sum( de i=1 a self.lambda_) [ fitness(Xi) * sum( de k=1 a N ) [ log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 ] ]
    #     total_loss /= self.nb_instances
    #     self.optimizerG.zero_grad()
    #     (-total_loss).backward()
    #     self.optimizerG.step()
    #     return total_loss


    # # Version REINFORCE
    def updateDistribution(self, solutionList, scoreList):
    #     # X => Individu ayant N variables {-1,1} [x1, x2, ..., xN] => exemple dans QUBO 64, chaque individu est une solution de 64 variables
    #     # X = solutions  # (nb_instances, λ, N)
    #     # i indice sur l'individu Xi [indi 1, indi 2, ..., indi λ]
    #     # k indice sur la variable de l'individu i [x1, x2, xK, xN] de l'individu Xi
    #     # n indice sur l'instance numero n
        device = self.device
        scoreList = scoreList.to(device)
        X = solutionList.to(device).squeeze(-1)  # (nb_instances, λ, N)

        L_Theta = 0.0
        if self.baseline is None:
            self.baseline = torch.zeros(self.nb_instances, device=device)

        for n in range(self.nb_instances):
            Pi_Theta = self.forward()[n]        # (N,)
            # on agrandit Pi_Theta sous la forme (λ, N) pour faire les opérations par lot
            # -1 signifie que la dimension N reste inchangée
            # unsqueeze(0) ajoute une dimension en position 0 pour la remplir avec λ
            Pi_Theta_expanded = Pi_Theta.unsqueeze(0).expand(self.lambda_, -1)  # (λ, N)
            actions = X[n]  # (λ, N)
            # f(Xi) pour chaque individu i de l'instance n
            fitness = scoreList[n]  # (λ,)

            # Log(Pi(Theta X)) = sum(log(Pi(Theta k)(a[k]))) = sum( log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 )
            Pi_selected = torch.where(actions == 1.0, Pi_Theta_expanded, 1.0 - Pi_Theta_expanded)  # (λ, N)
            log_Pi = torch.log(Pi_selected).sum(dim=1) 

            # L(θ) = moyenne des fitness moins la baseline pondérées par log π
            loss_n = torch.mean((fitness - self.baseline[n]) * log_Pi)

            self.optimizerG.zero_grad()
            (-loss_n).backward()
            self.optimizerG.step()

            self.baseline[n] = fitness.mean().item()

            L_Theta += loss_n
        # L(Theta) = (1/self.lambda_) * sum( de i=1 a self.lambda_) [ fitness(Xi) * sum( de k=1 a N ) [ log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 ] ]
        # On divise par le nombre d'instance pour un Theta global
        L_Theta /= self.nb_instances
        return L_Theta

    def toString(self):
        return "Strategy_Univariate_PPO_EDA number " + str(self.agent_number)

class MultiAgentUnivariateEDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - M agents travaillent sur toutes les instances
    - Budget λ divisé entre agents (λ/M solutions par agent)
    - Diversité via learning rates différents
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, M=4, device="cuda:0"):
        Abstract_EDA.__init__(self, N, lambda_, device)
        nn.Module.__init__(self)

        self.M = M
        self.N = N
        self.lambda_ = lambda_
        self.device = device

        self.lambda_per_agent = lambda_ // M
        remainder_lambda = lambda_ % M


        self.agents = nn.ModuleList()
        self.agent_lambdas = []
        bonus_indices = random.sample(range(M), remainder_lambda) if remainder_lambda > 0 else []
        
        for i in range(M):
            agent_lambda = self.lambda_per_agent + (1 if i in bonus_indices else 0)
            self.agent_lambdas.append(agent_lambda)
            
            agent = UnivariatePPOEDA(
                N, 
                agent_lambda, 
                beta, 
                typeModel, 
                dim_variables, 
                0.02,
                device, 
                agent_number=i
            )
            self.agents.append(agent)

    def reset_learned_parameters(self, nb_instances):
        self.nb_instances = nb_instances
        for i, agent in enumerate(self.agents):
            agent.reset_learned_parameters(nb_instances)  

    def sample_solutions(self):
        samples_list = []
        for agent in self.agents:
            samples = agent.sample_solutions()  # (nb_instances, λ_agent, N, 1)
            samples_list.append(samples)
        
        return torch.cat(samples_list, dim=1)  # (nb_instances, λ, N, 1)

    def updateDistribution(self, solutionList, scoreList):
        total_loss = 0.0
        start_lambda = 0

        for i, agent in enumerate(self.agents):
            agent_lambda = self.agent_lambdas[i]
            end_lambda = start_lambda + agent_lambda
            agent_solutions = solutionList[:, start_lambda:end_lambda, :, :]  # (nb_instances, λ_agent, N, 1)
            agent_scores = scoreList[:, start_lambda:end_lambda]              # (nb_instances, λ_agent)
            loss = agent.updateDistribution(agent_solutions, agent_scores)
            total_loss += loss

            start_lambda = end_lambda

        return total_loss / self.M

    def forward(self):
        return torch.stack([agent.forward() for agent in self.agents], dim=0)

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambda{self.lambda_}"