import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device utilisé : {device}")

N = 64
M = 15
nb_iter = 1000
beta = 0.1
lam = 200

from utils.walsh_expansion import WalshExpansion
puboi_path = f"source_code/instances/QUBO/puboi_evo_n_{N}_t_0_i_1.json"

try:
    we = WalshExpansion()
    we.load(puboi_path)
    Q_np = we.to_symmetric_Q()
    Q_th = torch.tensor(Q_np, dtype=torch.float32, device=device)
    print(f"✅ Instance PUBOi chargée : {puboi_path} (n={we.n})")
except Exception as e:
    Q_th = None
    print(f"⚠️ Impossible de charger {puboi_path} : {e}")

def qubo_reward(samples):
    if Q_th is None:
        return torch.zeros((samples.size(0),), device=samples.device)
    x = samples.float()
    s = 2.0 * x - 1.0
    Qs = torch.matmul(s, Q_th)
    energy = (Qs * s).sum(dim=1)
    return -energy

class UnivariatePPOEDA(nn.Module):
    def __init__(self, n_vars):
        super().__init__()
        self.linear = nn.Linear(n_vars, n_vars, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self, _=None):
        logits = self.linear.bias
        return torch.sigmoid(logits)

agents = [UnivariatePPOEDA(N).to(device) for _ in range(M)]
optimizers = [optim.Adam(agent.parameters(), lr=0.03 + 0.01 * i) for i, agent in enumerate(agents)]

for i, agent in enumerate(agents):
    agent.linear.bias.data = torch.randn_like(agent.linear.bias) * 0.5 + (i - M // 2)

fitness_history = [[] for _ in range(M)]

for t in range(nb_iter):
    all_fitness = []
    agent_probs_cache = []

    for i, (agent, opt) in enumerate(zip(agents, optimizers)):
        probs = agent()  # (N,)
        agent_probs_cache.append(probs.detach())

        dist = torch.distributions.Bernoulli(probs=probs)
        samples = dist.sample((lam,))  # (lam, N)

        fitness = qubo_reward(samples)
        sorted_idx = torch.argsort(fitness, descending=True)
        elite = samples[sorted_idx[:lam // 5]]
        target_probs = elite.mean(dim=0)

        ppo_loss = torch.mean((probs - target_probs) ** 2)

        old_probs = probs.detach()
        new_probs = probs
        kl = kl_divergence(
            torch.distributions.Bernoulli(probs=old_probs),
            torch.distributions.Bernoulli(probs=new_probs)
        ).mean()

        loss = ppo_loss + beta * kl

        opt.zero_grad()
        loss.backward()
        opt.step()

        all_fitness.append(fitness.mean().item())

    if t % 10 == 0:
        print(f"Iter {t:03d} | fitness = {[round(f,2) for f in all_fitness]}")

    for i in range(M):
        fitness_history[i].append(all_fitness[i])

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
for i in range(M):
    plt.plot(fitness_history[i], label=f"Agent {i+1}")
plt.xlabel("Itération")
plt.ylabel("Fitness moyenne")
plt.title("Évolution de la fitness par agent")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
