import torch
import torch.nn as nn

from .utils import adaptative_bandwith

class JSD(nn.Module):
    def __init__(self, bandwith_kernel=None, tau=10000.0):
        super().__init__()
        self.bandwith_kernel = bandwith_kernel
        self.tau = float(tau)

    def forward(self, Thetas, probs=None):
        if probs is None:
            raise ValueError("JSD kernel requires probs.")
        Thetas = Thetas.requires_grad_(True)
        if Thetas.dim() == 4:
            B, M, N, D = Thetas.shape
        else:
            B, M, N = Thetas.shape

        eps = 1e-7
        if Thetas.dim() == 4:
            pi = probs.unsqueeze(2)                              
            pj = probs.detach().unsqueeze(1)                     
            m = 0.5 * (pi + pj)

            pi_c = pi.clamp(min=eps)
            pj_c = pj.clamp(min=eps)
            m_c = m.clamp(min=eps)

            kl_pm = (pi * torch.log(pi_c / m_c)).sum(dim=-1)                   
            kl_qm = (pj * torch.log(pj_c / m_c)).sum(dim=-1)                   

            jsd = 0.5 * (kl_pm + kl_qm)                                        
            dist = jsd.sum(dim=-1) / float(N)                                         
        else:
            pi = probs.unsqueeze(2)                           
            pj = probs.detach().unsqueeze(1)                  
            m = 0.5 * (pi + pj)

            pi_c = pi.clamp(min=eps, max=1.0 - eps)
            pj_c = pj.clamp(min=eps, max=1.0 - eps)
            m_c = m.clamp(min=eps, max=1.0 - eps)

            kl_pm = pi * torch.log(pi_c / m_c) + (1.0 - pi) * torch.log((1.0 - pi_c) / (1.0 - m_c))
            kl_qm = pj * torch.log(pj_c / m_c) + (1.0 - pj) * torch.log((1.0 - pj_c) / (1.0 - m_c))

            jsd = 0.5 * (kl_pm + kl_qm)                                        
            dist = jsd.sum(dim=-1) / float(N)                                         

        if self.bandwith_kernel is None:
            gamma = adaptative_bandwith(dist, eps=1e-3)
        else:
            gamma = self.bandwith_kernel

        tau2 = self.tau ** 2
        if Thetas.dim() == 4:
            probs_c = probs.clamp(min=eps, max=1.0 - eps)
            g = torch.exp(-1.0 / (tau2 * probs_c * (1.0 - probs_c)) + (4.0 / tau2))                
            mask = getattr(self, "mask", None)
            if mask is not None:
                mask = mask.to(probs.device, probs.dtype)
                g = g * mask + (1.0 - mask)
            g_var = g.prod(dim=-1)             
            prod_g = g_var.prod(dim=-1)          
        else:
            probs_c = probs.clamp(min=eps, max=1.0 - eps)
            g = torch.exp(-1.0 / (tau2 * probs_c * (1.0 - probs_c)) + (4.0 / tau2))             
            prod_g = g.prod(dim=-1)          
        g_pair = prod_g.unsqueeze(2) * prod_g.unsqueeze(1)             

        K = torch.exp(-gamma * dist) * g_pair                           

        if Thetas.dim() == 4:
            grad_Thetas = torch.zeros_like(Thetas)
            for i in range(M):
                Ki = K[:, :, i]
                vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
                grad_Thetas[:, i, :, :] = torch.sum(vect_grad_Thetas, dim=1)
        else:
            grad_Thetas = torch.zeros((B, M, N), device=Thetas.device, dtype=Thetas.dtype)
            for i in range(M):
                Ki = K[:, :, i]
                vect_grad_Thetas, = torch.autograd.grad(Ki.sum(), Thetas, retain_graph=True)
                grad_Thetas[:, i, :] = torch.sum(vect_grad_Thetas, dim=1)
        return K, grad_Thetas
