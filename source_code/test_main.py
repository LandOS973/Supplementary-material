import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device utilisé : {device}")

N = 3
B = 2
L = 5

torch_test_tensor = torch.randn((B, N), device=device)
print("tensor print test : ", torch_test_tensor)

torch_test_expand = torch_test_tensor.unsqueeze(-1)
print("tensor unsqueeze test : ", torch_test_expand, print(torch_test_expand.shape))

scorelist = torch.randn((B, L), device=device)
print("scorelist test : ", scorelist, print(scorelist.shape))

scorelist_squeezed = scorelist.squeeze(-1)
print("scorelist squeezed test : ", scorelist_squeezed, print(scorelist_squeezed.shape))