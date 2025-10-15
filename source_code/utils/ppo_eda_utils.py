import math
import torch
from torch.nn import Parameter
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from typing import Optional, Any
from torch import Tensor
from torch.nn import init
from utils.pl import PL, NeuralSort


class OrderGenerator(torch.nn.Module):
    
    def __init__(self, device, nb_instances, size_pop, N):
        super(OrderGenerator, self).__init__()

        self.N = N
        self.nb_instances = nb_instances
        self.size_pop = size_pop
        self.device = device

        self.A = torch.tensor(np.triu(np.ones((self.nb_instances, size_pop, self.N, self.N)), k=1)).to(self.device).float()

        self.knownP = None
        self.knownDAG = None
        


            
    def setKnownDAG(self, knownDAG):
        
        self.knownDAG = knownDAG
        
        
    def setKnownOrder(self, knownOrder):
 
        self.knownOrder = knownOrder
        self.knownP = torch.nn.functional.one_hot(knownOrder, num_classes = self.N).float()
        

    def get_order(self, random):

        with torch.no_grad():
    
            if(self.knownP != None and random ==False):
                order = self.knownOrder
                mask = torch.transpose(self.knownP, 2, 3) @ self.A.float() @ self.knownP
            else:
                
                order = torch.argsort(torch.rand(self.nb_instances, self.size_pop, self.N)).to(self.device)
                P = torch.nn.functional.one_hot(order, num_classes = self.N).float()
                mask = torch.transpose(P, 2, 3) @ self.A.float() @ P
                
                
            if(self.knownDAG is not None):
                mask = mask*self.knownDAG
                
        return order, mask

def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    #U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)
    return torch.log(U) - torch.log(1-U)


def _sigmoid_sample(logits, tau=1, sample=True):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())

    if(sample):
        y = logits + logistic_noise
    else:
        y = logits

    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits,  ones_tensor, zeros_tensor, tau=1, hard=False, sample=True):

    shape = logits.size()


    y_soft = _sigmoid_sample(logits, tau=tau, sample=sample)


    if hard:
        y_hard = torch.where(y_soft > 0.5, ones_tensor, zeros_tensor)
        y = y_hard.data - y_soft.data + y_soft
    else:
    	y = y_soft


    return y

class MatrixSampler(torch.nn.Module):
    """Matrix Sampler, following a Bernoulli distribution. Differenciable."""
    def __init__(self, nb_trajectories, size_pop, graph_size, mask=None, gumble=False):
        super(MatrixSampler, self).__init__()
        if not isinstance(graph_size, (list, tuple)):
            self.graph_size = (nb_trajectories, graph_size, graph_size)
        else:
            self.graph_size = graph_size

        self.size_pop = size_pop

        self.weights = torch.nn.Parameter(torch.FloatTensor(*self.graph_size))

        self.weights.data.zero_()


        if mask is None:
            mask = 1 - torch.eye(self.graph_size[1],self.graph_size[2])


        self.register_buffer("mask", mask)


        self.gumble = gumble


        self.nb_trajectories = nb_trajectories

        ones_tensor = torch.ones(*self.graph_size)
        ones_tensor = ones_tensor.repeat(self.size_pop, 1,  1)

        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = torch.zeros(*self.graph_size)
        zeros_tensor = zeros_tensor.repeat(self.size_pop, 1,  1)
        self.register_buffer("zeros_tensor", zeros_tensor)


    def updateMask(self, mask):


        self.register_buffer("mask", mask)



    def forward(self, tau=1, drawhard=True, sample=True):
        """Return a sampled graph."""

        # if(self.gumble):
        #
        #     drawn_proba = gumbel_softmax(torch.stack([self.weights.unsqueeze(1).repeat(1,self.size_pop, 1,  1).view(-1), -self.weights.expand(self.size_pop, 1, 1, 1).view(-1)], 1),
        #                        tau=tau, hard=drawhard)[:, 0].view(*self.graph_size)
        # else:
        drawn_proba = gumbel_sigmoid(self.weights.unsqueeze(1).repeat(1,self.size_pop, 1,  1).view(self.nb_trajectories*self.size_pop,self.graph_size[1], self.graph_size[2]), self.ones_tensor, self.zeros_tensor, tau=tau, hard=drawhard, sample=sample)



        drawn_proba = drawn_proba.view(self.nb_trajectories, self.size_pop, self.graph_size[1], self.graph_size[2])

        test = self.mask * drawn_proba


        return  test


    def get_proba(self):
        return torch.sigmoid( self.weights) * self.mask
        # if hasattr(self, "mask"):
        #     return torch.sigmoid(2 * self.weights) * self.mask
        # else:
        #     return torch.sigmoid(2 * self.weights)

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)

    def reset_parameters(self, init_value):

        self.weights.data.fill_(init_value)


class LearnedOrderGenerator(torch.nn.Module):
    def __init__(self, device,  nb_instances, size_pop, N,  noise_rescale ):
        super(LearnedOrderGenerator, self).__init__()


        self.N = N
        self.nb_instances = nb_instances
        self.size_pop = size_pop

        self.device = device
        
        self.weights = torch.nn.Parameter(torch.FloatTensor(nb_instances, N))

        self.A = torch.tensor(np.triu(np.ones((nb_instances, size_pop, N, N)), k=1)).to(device).float()

        self.soft_sort = NeuralSort(self.device,tau=1, hard=True)

        self.order_num = torch.tensor(np.arange(self.N)).to(device).float()
        
        self.noise_rescale = noise_rescale

    def get_order(self, _):
        
        
        pl_s = PL(self.device, self.weights**2+0.01, 1, self.noise_rescale, hard=True)
        P_hat = pl_s.sample((self.nb_instances,self.size_pop,))

        sorted_index = P_hat @ self.order_num
        mask = torch.transpose(P_hat, 2, 3) @ self.A.float() @ P_hat

        return sorted_index, mask

    def reset_parameters(self):
        
        #self.weights.data.uniform_(0,self.amplitude)
        self.weights.data.fill_(1)




class LinearCustom(torch.nn.Module):

    def __init__(self, nb_instances, channels, in_features, out_features, size_pop,  batch_size=-1, bias=True):
        super(LinearCustom, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        
        self.size_pop = size_pop
        
        self.weight = Parameter(torch.Tensor(nb_instances, channels, self.in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(nb_instances, channels, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self,  data, order_variable=None):


        if(order_variable is not None):

            all_weights = self.weight.unsqueeze(1).repeat(1,self.size_pop,1,1,1)
            selected_weights = torch.gather(all_weights, 2, order_variable.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,1,all_weights.size()[3],all_weights.size()[4])).squeeze(2)
            all_bias = self.bias.unsqueeze(1).repeat(1, self.size_pop, 1, 1)
            selected_bias = torch.gather(all_bias, 2, order_variable.unsqueeze(2).unsqueeze(3).repeat(1,1,1,all_bias.size()[3])).squeeze(2)

            output = (data.unsqueeze(2)@selected_weights).squeeze(2).squeeze(-1)
            output = output  + selected_bias.squeeze()

        else:

            test_weights = self.weight.unsqueeze(1)

            output = (data.unsqueeze(3) @ test_weights).squeeze(3)
            test_bias = self.bias.unsqueeze(1)

            output = (output + test_bias).squeeze(-1)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )




class PPO_EDA_generator(torch.nn.Module):
    """Ensemble of all the generators."""


    def __init__(self, data_shape, nh, size_pop, skeleton=None, cat_sizes=None, linear=False, numberHiddenLayersG=1, device="cuda:0"):
        """Init the model."""
        super(PPO_EDA_generator, self).__init__()
        layers = []

        self.sizes = cat_sizes
        self.linear = linear

        nb_vars = data_shape[2]

        self.nb_vars = nb_vars
        self.batch_size = data_shape[0]
        self.activation = torch.nn.Tanh()

        if cat_sizes is not None:

            self.max_cat_size = max(cat_sizes)
            size_data_input = self.max_cat_size * self.nb_vars
            output_dim = self.max_cat_size

        else:
            output_dim = 1
            size_data_input = nb_vars

        self.device = device


        if linear:

            self.input_layer = LinearCustom(data_shape[0], nb_vars, size_data_input, output_dim, size_pop)
        else:

            self.input_layer = LinearCustom(data_shape[0], nb_vars, size_data_input, nh, size_pop)
            
            self.list_hidden_layer = []
            
            for i in range(numberHiddenLayersG):
                self.list_hidden_layer.append( LinearCustom(data_shape[0],nb_vars, nh, nh, size_pop))
            
            self.output_layer = LinearCustom(data_shape[0],nb_vars, nh, output_dim, size_pop)


    def forward(self,  data, mask_input, mask_output,  order_variables=None):


        if(order_variables is not None):

            if self.sizes is not None:

                data_input_tmp = torch.nn.functional.one_hot(data.long(), self.max_cat_size).float() * 2 -1
                data_input = data_input_tmp * mask_input.unsqueeze(3)
                data_input = data_input.view(data_input.size()[0], data_input.size()[1], -1)
                
            else:
                data_input = data * mask_input* 2 - 1

            if self.linear:
                output = self.input_layer(data_input, order_variables)
            else:

                out = self.input_layer(data_input, order_variables)
                out = self.activation(out)
                
                for hidden_layer in self.list_hidden_layer:
                    out = hidden_layer(out, order_variables)
                    out = self.activation(out)
                    
                output = self.output_layer(out,order_variables)

        else:

            data = data.unsqueeze(2).repeat(1, 1, self.nb_vars, 1)

            if self.sizes is not None:

                data_input_tmp = torch.nn.functional.one_hot(data.long(), self.max_cat_size).float() * 2 -1
                data_input = data_input_tmp * mask_input.unsqueeze(4)
                data_input = data_input.view(data_input.size()[0], data_input.size()[1], data_input.size()[2], -1)

            else:

                data_input = data * mask_input* 2 -1

            if self.linear:
                output = self.input_layer(data_input)
            else:
                out = self.input_layer(data_input)
                out = self.activation(out)
                
                for hidden_layer in self.list_hidden_layer:
                    out = hidden_layer(out)
                    out = self.activation(out)
                    
                output = self.output_layer( out)

        if self.sizes is not None:

            if(mask_output  is not None):
                output = output + mask_output
                
            output = torch.softmax(output ,  -1)

        else:
            output = torch.sigmoid(output)

        return output


    def reset_parameters(self):
        if not self.linear:
            self.output_layer.reset_parameters()
            
            for hidden_layer in self.list_hidden_layer:
                hidden_layer.reset_parameters()
                hidden_layer.to(self.device)
                
        self.input_layer.reset_parameters()

