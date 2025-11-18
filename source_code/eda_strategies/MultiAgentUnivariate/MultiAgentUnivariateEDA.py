import torch
import torch.nn as nn
from eda_strategies.Abstract_EDA import Abstract_EDA
import random
from torch.distributions import Bernoulli, kl_divergence as torch_kl_divergence
from . import SVGD, rbf

class UnivariatePPOEDA(Abstract_EDA, nn.Module):
    """
    Version univariée propre : chaque variable Xi est indépendante, 
    p(Xi,j=1) = sigmoid(theta_i_j) avec theta_i_j appris via une couche linéaire.
    i = instance, j = variable
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, learning_rate, device, agent_number, update_method, K_steps, beta_adapt, delta_target):
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
        self.opt_reinforce = None
        self.opt_ppo = None
        self.register_buffer("beta_vector", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("baseline",    torch.empty(0, dtype=torch.float32), persistent=False)
        self.agent_number = agent_number
        self.theta_old = None
        self.update_method = update_method
        self.K_steps = K_steps
        self.beta_adapt = beta_adapt
        self.delta_target = delta_target
        self.register_buffer("grad_rl", torch.empty(0, dtype=torch.float32), persistent=False)

    def forward(self):
        """
        Retourne les probabilités p(x_i=1) = sigmoid(theta_i)
        theta_i sont les poids de la couche linéaire.
        """
        probs = torch.sigmoid(self.theta)
        return probs  # (B, N)

    def reset_learned_parameters(self, nb_instances):
        """Réinitialise les paramètres et sauvegarde le nombre d'instances."""
        self.theta = nn.Parameter(torch.zeros((nb_instances, self.N), device=self.device)) # (B, N)
        self.theta_old = self.theta.detach().clone()
        self.nb_instances = nb_instances # B
        self.beta_vector.resize_(nb_instances).fill_(1.0)
        self.grad_rl.resize_(nb_instances, self.N).zero_()
        self.baseline.resize_(nb_instances).zero_()
        self.opt_reinforce = torch.optim.SGD(
            [self.theta], lr=self.learning_rate
        )
        self.opt_ppo = torch.optim.Adam(
            [self.theta], lr=self.learning_rate
        )

    def sample_solutions(self):
        probs = torch.sigmoid(self.theta)                     # (B, N)
        probs = torch.clamp(torch.nan_to_num(probs, nan=0.5), 1e-6, 1 - 1e-6)
        u = torch.rand((self.nb_instances, self.lambda_, self.N), device=self.device)
        samples = (u < probs.unsqueeze(1)).float().unsqueeze(-1)   # (B, λ, N, 1)
        return samples

    
    def kl_divergence(self, p, q):
        eps = 1e-7
        p = torch.clamp(torch.nan_to_num(p, nan=0.5), eps, 1 - eps)
        q = torch.clamp(torch.nan_to_num(q, nan=0.5), eps, 1 - eps)
        return p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
    
    def updateDistributionPPO(self, solutionList, scoreList):
        B = self.nb_instances
        N = self.N # nombre de variables
        λ = self.lambda_
        # ratio = torch.exp(log(πθ) - log(πθk))
        # log(πθ(a|s)) = sum( log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 )
        device = self.device
        scoreList = scoreList                 # (B, λ)
        K_steps = self.K_steps
        total_loss = 0.0
        with torch.no_grad():
            # capture de πθk
            p_old = torch.sigmoid(self.theta).clamp(1e-10, 1-1e-10)    # (B, N)
        # Données batchées
        Dk = solutionList.squeeze(-1)                        # (B, λ, N) => solutionList est (B, λ, N, 1)
        fitnesses = scoreList  
        if self.baseline is None:
            self.baseline = torch.zeros(B, device=device) # (B,)
        adv = fitnesses - self.baseline.unsqueeze(1)          # (B, λ)
        adv = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True) + 1e-10)
        # log πθk(a|s)
        p_old_exp = p_old.unsqueeze(1).expand(-1, λ, -1) # (B, λ, N) => pour comparer avec Dk
        # log πθk(a|s) = ∑ k=1 a n de log(πθk(a k|s)) => ak est la valeur de la variable k dans l'action a (la variable k sur l'individu λi)  # (B, λ) 
        # πθk(a k|s) = Sigmoid(θk) si ak=1 et 1-Sigmoid(θk) si ak=0
        log_pi_old = torch.where(
            Dk == 1.0,
            torch.log(p_old_exp + 1e-10),
            torch.log(1.0 - p_old_exp + 1e-10) # (B, λ, N)
        ).sum(dim=2) # (B, λ)                                               
        for _ in range(K_steps):
            # πθ(a|s)
            p_new = torch.sigmoid(self.theta).clamp(1e-10, 1-1e-10)     # (B, N)
            p_new_exp = p_new.unsqueeze(1).expand(-1, λ, -1)            # (B, λ, N)
            # log πθ(a|s) et ratio
            log_pi_new = torch.where(
                Dk == 1.0,
                torch.log(p_new_exp + 1e-10),
                torch.log(1.0 - p_new_exp + 1e-10)
            ).sum(dim=2)                        # (B, λ)
            ratio = torch.exp(log_pi_new - log_pi_old)              # (B, λ)
            # mean (exp(log πθ - log πθk) * adv(s,a))
            L_per_i = (ratio * adv).mean(dim=1)  # (B,) moyenne sur les λ actions
            # DKL(πθk (.|s)|πθ(.|s)).
            KL_mean_i = self.kl_divergence(p_old, p_new).mean(dim=1) # (B,)
            # ∇θ - L_{θk}(θ) + β_k ∇θ D̄_KL(θk || θ) (minimization)
            if self.beta_adapt is True:
                loss_i = -L_per_i + self.beta_vector * KL_mean_i                   # (B,)
            else:
                loss_i = -L_per_i + self.beta * KL_mean_i                   # (B,)
            # Addition des losses descentes de gradient individuelles i
            loss = loss_i.sum()
            self.opt_ppo.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                self.grad_rl.copy_(-self.theta.grad)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([self.theta], max_norm=1.0)
            self.opt_ppo.step()
            total_loss += loss.item()
        with torch.no_grad():
            p_new_final = torch.sigmoid(self.theta).clamp(1e-10, 1-1e-10)
        if self.beta_adapt is True:
            # if D¯KL(θk |θk+1) ≥ 1.5δ then
            #   βk+1 ← 2βk
            # else if D¯KL(θk |θk+1) ≤ δ/1.5 then
            #   βk+1 ← 0.5βk
            # version with masks
            with torch.no_grad():
                KL_final_i = self.kl_divergence(p_old, p_new_final).mean(dim=1)  # (B,)
            mask_up = KL_final_i >= 1.5 * self.delta_target
            mask_down = KL_final_i <= (self.delta_target / 1.5)
            self.beta_vector[mask_up]   *= 2.0
            self.beta_vector[mask_down] *= 0.5
            # clamping
            self.beta_vector = torch.clamp(self.beta_vector, 1e-4, 10)
        return total_loss / self.nb_instances


    # # Version REINFORCE
    def updateDistributionREINFORCE(self, solutionList, scoreList):
    #     # X => Individu ayant N variables {-1,1} [x1, x2, ..., xN] => exemple dans QUBO 64, chaque individu est une solution de 64 variables
    #     # X = solutions  # (nb_instances, λ, N)
    #     # i indice sur l'individu Xi [indi 1, indi 2, ..., indi λ]
    #     # k indice sur la variable de l'individu i [x1, x2, xK, xN] de l'individu Xi
    #     # n indice sur l'instance numero n
        device = self.device
        scoreList = scoreList
        actions = solutionList.squeeze(-1)  # (nb_instances, λ, N)
        if self.baseline is None:
            self.baseline = torch.zeros(self.nb_instances, device=device) # (nb_instances,)
        all_Pi_Theta = self.forward()  # (nb_instances, N)
        all_Pi_Theta_expanded = all_Pi_Theta.unsqueeze(1).expand(-1, self.lambda_, -1)  # (nb_instances, λ, N)
        fitness = scoreList  # (nb_instances, λ)
        # Log(Pi(Theta X)) = sum(log(Pi(Theta k)(a[k]))) = sum( log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 )
        Pi_selected = torch.where(actions == 1.0, all_Pi_Theta_expanded, 1.0 - all_Pi_Theta_expanded)  # (nb_instances, λ, N)
        log_Pi = torch.log(Pi_selected + 1e-10).sum(dim=2)  # (nb_instances, λ)
        advantages = (fitness - self.baseline.unsqueeze(1))  # (nb_instances, λ)
        # L(Theta) = (1/self.lambda_) * sum( de i=1 a self.lambda_) [ fitness(Xi) * sum( de k=1 a N ) [ log(Sigmoid(Theta k)) si a[k]=1 et log(1-Sigmoid(Theta k)) si a[k]=-1 ] ]
        # L(θ) = moyenne des fitness moins la baseline pondérées par log π
        loss_per_instance = torch.mean(advantages * log_Pi, dim=1)  # (nb_instances,)
        loss = -loss_per_instance.sum() 
        self.opt_reinforce.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            self.baseline = fitness.mean(dim=1)  # (nb_instances,)
            self.grad_rl.copy_(-self.theta.grad)
        self.opt_reinforce.step()
        return loss_per_instance.mean()

    def updateDistribution(self, solutionList, scoreList):
        """Dispatch to PPO or REINFORCE depending on the agent flag."""
        if self.update_method == "PPO":
            return self.updateDistributionPPO(solutionList, scoreList)
        else:
            return self.updateDistributionREINFORCE(solutionList, scoreList)

    def toString(self):
        return "Strategy_Univariate_PPO_EDA number " + str(self.agent_number)

class MultiAgentUnivariateEDA(Abstract_EDA, nn.Module):
    """
    Multi-agent collaboratif :
    - M agents travaillent sur toutes les instances
    - Budget λ divisé entre agents (λ/M solutions par agent)
    - Diversité via learning rates différents
    """

    def __init__(self, N, lambda_, beta, typeModel, dim_variables, M, device, updateMethod, K_steps, beta_adapt, delta_target, learning_rate):
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
                N, agent_lambda, beta, typeModel, dim_variables,
                learning_rate,
                device, agent_number=i, update_method=updateMethod,
                K_steps=K_steps, beta_adapt=beta_adapt, delta_target=delta_target
            ).to(device)
            self.agents.append(agent)

        # SVGD coupling between agents: kernel and step size
        # Default step size can be tuned; keep small by default
        self.svgd = SVGD.SVGD(rbf.RBF())
        self.eta_svgd = 0.001

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
        try:
            grads = torch.stack([agent.grad_rl for agent in self.agents], dim=0)  # (M, B, N)
        except Exception:
            grads = None

        if grads is not None:
            M = len(self.agents)
            B = self.nb_instances
            device = self.device
            # For each instance b, compute phi over agents and apply update to each agent.theta[b]
            for b in range(B):
                # build theta matrix (M, N) and score matrix (M, N)
                theta_mat = torch.stack([agent.theta.data[b].detach() for agent in self.agents], dim=0).to(device)
                score_mat = grads[:, b, :].to(device)
                # compute SVGD direction phi (M, N)
                phi = self.svgd.phi(theta_mat, score_mat)  # returns detached tensor (M, N)
                with torch.no_grad():
                    for a in range(M):
                        self.agents[a].theta.data[b].add_(self.eta_svgd * phi[a].to(self.agents[a].theta.device))

        return total_loss / self.M
    
    def get_score_b(self):
        return torch.stack([agent.grad_rl for agent in self.agents], dim=0)  # (M, nb_instances, N)
    
    
    


    def forward(self):
        return torch.stack([agent.forward() for agent in self.agents], dim=0)

    def toString(self):
        return f"MultiAgent_Collaborative_M{self.M}_lambda{self.lambda_}"