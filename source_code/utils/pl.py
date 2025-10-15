'''
Relaxed Plackett-Luce distribution.
'''

from numbers import Number

import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.distributions import constraints

# use GPU if available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


class PL(Distribution):

    arg_constraints = {'scores': constraints.positive,
                       'tau': constraints.positive}
    has_rsample = True

    @property
    def mean(self):
        # mode of the PL distribution
        return self.relaxed_sort(self.scores)

    def __init__(self, device, scores, tau, noise_rescale, hard=True, validate_args=None):
        """
        scores. Shape: (batch_size x) n 
        tau: temperature for the relaxation. Scalar.
        hard: use straight-through estimation if True
        """
        self.scores = scores.unsqueeze(-1)
        self.tau = tau
        self.hard = hard
        self.n = self.scores.size()[1]

        self.noise_rescale = noise_rescale
        self.device = device

        if isinstance(scores, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scores.size()

        super(PL, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args:
            if not torch.gt(self.scores, torch.zeros_like(self.scores)).all():
                raise ValueError("PL is not defined when scores <= 0")

    def relaxed_sort(self, inp):
        """
        inp: elements to be sorted. Typical shape: batch_size x n x 1
        """
        bsize = inp.size()[0]
        dim = inp.size()[1]
        one = FloatTensor(dim, 1, device = self.device).fill_(1)


        A_inp = torch.abs(inp - inp.permute(0, 2, 1))
        B = torch.matmul(A_inp, torch.matmul(one, torch.transpose(one, 0, 1)))

        scaling = (dim + 1 - 2 * (torch.arange(dim, device = self.device) + 1)).type(FloatTensor)
        C = torch.matmul(inp, scaling.unsqueeze(0))
        P_max = (C-B).permute(0, 2, 1)

        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device = self.device)

            b_idx = torch.arange(bsize, device = self.device).repeat([1, dim]).view(
                dim, bsize).transpose(dim0=1, dim1=0).flatten().type(LongTensor)

            r_idx = torch.arange(dim, device = self.device).repeat(
                [bsize, 1]).flatten().type(LongTensor)


            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda

            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1

            P_hat = (P-P_hat).detach() + P_hat

        return P_hat

    def rsample(self, sample_shape, log_score=True):
        """
        sample_shape: number of samples from the PL distribution. Scalar.
        """


        with torch.enable_grad():  # torch.distributions turns off autograd

            n_samples = sample_shape[0] * sample_shape[1]

            def sample_gumbel(device, samples_shape, eps=1e-20):

                U = torch.zeros(samples_shape, device = device).uniform_()

                return -torch.log(-torch.log(U + eps) + eps)

            if not log_score:


                test = sample_gumbel(self.device, [n_samples, 1, self.n, 1])

                log_s_perturb = torch.log(self.scores.unsqueeze(0) + test)



            else:


                log_s_perturb = self.scores.unsqueeze(1).repeat(1,sample_shape[1], 1, 1).view(sample_shape[0] * sample_shape[1],self.n,1 ) + sample_gumbel(self.device, [n_samples, self.n, 1])/self.noise_rescale


            P_hat = self.relaxed_sort(log_s_perturb)

 
            P_hat = P_hat.view(sample_shape[0],sample_shape[1], self.n, self.n)



            return P_hat.squeeze()

    def log_prob(self, value):
        """
        value: permutation matrix. shape: batch_size x n x n
        """
        permuted_scores = torch.squeeze(torch.matmul(value, self.scores))
        log_numerator = torch.sum(torch.log(permuted_scores), dim=-1)
        idx = LongTensor([i for i in range(self.n-1, -1, -1)])
        invert_permuted_scores = permuted_scores.index_select(-1, idx)
        denominators = torch.cumsum(invert_permuted_scores, dim=-1)
        log_denominator = torch.sum(torch.log(denominators), dim=-1)
        return (log_numerator - log_denominator)



from torch import Tensor

class NeuralSort (torch.nn.Module):
    def __init__(self,device, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.device = device

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = torch.FloatTensor(dim, 1).fill_(1).to(self.device)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(torch.FloatTensor).to(self.device)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=self.device)
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.LongTensor).to(self.device)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(torch.LongTensor).to(self.device)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat

        return P_hat



if __name__ == '__main__':

    device = "cuda:0"

    scores = torch.Tensor([[100.8, 0.3, 11111.9]]).unsqueeze(-1).to(device)
    tau = 0.1

    # hard = True is necessary
    pl_dist = PL(scores, tau, hard=False)

    sorted_scores = pl_dist.relaxed_sort(scores)

    print("sorted_scores")
    print(sorted_scores)
    print(sorted_scores.size())



    #
    # # check helper sorting function
    # sorted_scores = pl_dist.relaxed_sort(scores)
    #
    # print("sorted_scores")
    # print(sorted_scores)
    # print(sorted_scores.size())
    #
    # # check if we get mode of distribution
    # # print(pl_dist.mean)
    #
    # # check log prob function
    # good_pm = torch.Tensor([[[0., 0., 1.],
    #                          [1., 0., 0.],
    #                          [0., 1., 0.]]])
    # intermediate_pm = torch.Tensor([[[0., 0., 1.],
    #                                  [0., 1., 0.],
    #                                  [1., 0., 0.]]])
    # bad_pm = torch.Tensor([[[0., 1., 0.],
    #                         [1., 0., 0.],
    #                         [0., 0., 1.]]])
    # print(pl_dist.log_prob(good_pm), pl_dist.log_prob(
    #     intermediate_pm), pl_dist.log_prob(bad_pm))
    # print()
    #
    # # check sample
    # scores_bimodal = torch.Tensor([[11111.92, 0.3, 11111.9]]).unsqueeze(-1)
    # pl_dist_bimodal = PL(scores_bimodal, tau, hard=True)
    # samples = pl_dist_bimodal.sample((5,))
    # print(samples)
    # print()
    #
    # # code for kl(q, p)
    # scores_prior = torch.Tensor([[0.3, 10.8, 1111.9]]).unsqueeze(-1)
    # tau_prior = 0.1
    #
    # pl_dist_prior = PL(scores_prior, tau_prior, hard=True)
    # print(pl_dist_prior.mean)
    # print(pl_dist_prior.log_prob(good_pm), pl_dist_prior.log_prob(
    #     intermediate_pm), pl_dist_prior.log_prob(bad_pm))
    #
    # # kl (q, p)
    # empirical_kl = pl_dist.log_prob(good_pm) - pl_dist_prior.log_prob(good_pm)
    # print(empirical_kl)
