#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Create one instance of PUBOi benchmark



@authors: LEC group, LISIC Lab, Univ. of Littoral Opal Coast, France
Created on Fri Apr 30 14:57:22 2021

Please cite this article if you use this code:
  Sara Tari, Sebastien Verel, and Mahmoud Omidvar. 
  "PUBOi: A Tunable Benchmark with Variable Importance." 
  In European Conference on Evolutionary Computation in Combinatorial Optimization (Part of EvoStar), pp. 175-190. Springer, Cham, 2022.

"""
import numpy as np
import scipy.special
from utils import walsh_expansion as we


class PUBOi_generator:
	def __init__(self, m, importance, factor, portfolio, p_function, typeWeight, shift, seed):
		# pb dimension
		self.n = sum(importance["size"])
		# number of terms (clauses)
		self.m = m 
		# information of importance of each class: degree of importance, and number of variable for each class
		self.importance = importance
		# number of class of importance
		self.n_class = len(self.importance["degree"])
		# factor over the independance probability to have the same of importance
		self.factor = factor
		# portfolio of basis functions
		self.portfolio = portfolio
		# probability of each portfolio function
		self.p_function = p_function
		# number of functions in the portfolio
		self.nf = len(self.p_function)
		# type of weights. 0: uniform, 1: geometric mean of importance
		self.typeWeight = typeWeight
		# apply random xor on each clause to shuffle the maximum of portfolio functions 
		self.shift = shift
		# seed of the random generator
		self.seed = seed
		
		# normalize to 1
		s = sum(self.importance["degree"])
		self.importance["pclass"] = [ (float(i) / s) for i in importance["degree"]]

		# Iterative probability to select one class
		for i in range(self.n_class - 1):
			for j in range(i + 1, self.n_class):
				self.importance["pclass"][j] = self.importance["pclass"][j] / (1 - self.importance["pclass"][i])
		
		# Probability to select (c, c) with c a class index
		self.pprime = [ self.factor * self.importance["pclass"][i] for i in range(self.n_class) ]

		s = sum(self.p_function)
		self.p_function = [float(i) / s for i in self.p_function]
	

	def make(self):
		# set seed
		np.random.seed(self.seed)

		number_class_function = self.sample_class()

		# importance class of each binary variable, and inverse, the list of variable for each importance class
		imp_var = self.create_importance_var()

		clauses = self.create_walsh(imp_var, number_class_function)

		self.objective = "min" # minimization problem

		f = we.WalshExpansion(self.n)
		self.bound = 0
		for clause in clauses:
			p = self.portfolio[ clause[0] ].make(clause[2])
			if self.shift:
				x = []
				for i in range(self.n):
					if np.random.random() < 0.5:
						x.append(1)
					else:
						x.append(-1)
				p.xor(x)

			p.mult(clause[1])
			f.sum(p)
			self.bound += clause[1] * self.portfolio[ clause[0] ].optimum

		f.simplify() # remove terms with value 0
		return f

	# Sample the number of function of each class function of the portfolio
	def sample_class(self):
		n_class = []

		s = np.random.choice(np.arange(0, self.nf), size = self.m, replace = True, p = self.p_function)

		s_list = s.tolist()
		for i in range(self.nf):
			n_class.append(s_list.count(i))
		
		return n_class

	# define the importance of each variable, and the variables set of each class
	def create_importance_var(self):
		vars_in_class = [ ]
		
		var_importance = [ 0 ] * self.n
		
		cumul_size = 0
		for i in range(self.n_class):
			# notice that vars id could be shuffled
			vars_in_class.append( [ j + cumul_size for j in range(self.importance["size"][i])] )
			
			for k in vars_in_class[i]:
				var_importance[ k ] = i 
			
			cumul_size = cumul_size + self.importance["size"][i]

		d = {"vars_in_class" : vars_in_class, "var_importance" : var_importance}

		return d

	def create_walsh(self, imp_var, number_class_function):
		W = []

		for k in range(self.nf):
			if (number_class_function[k] > 0):
				terms = self.create_terms(imp_var, self.portfolio[k].arity, number_class_function[k])
				weights = self.create_weigths(imp_var, terms)

				for (w, t) in zip(weights, terms):
					W.append( (k, w, t) )

		return(W)

	def create_terms(self, imp_var, arity, n_clauses) :
		probV = [ [] ]
		for n in range(1, arity + 1):
			probc = []
			for c in range(self.n_class - 1):
				prob = []
				cumul = 0.0
				for i in range(n):
					pr = scipy.special.binom(n, i) * (1 - self.pprime[c])**i * self.pprime[c]**(n - 1 - i) * self.importance["pclass"][c]
					#print(n, i, pr)
					cumul = cumul + pr
					if cumul <= 1:
						prob.append(cumul)
					else:
						print("error: factor is too small, or too large: cumul=" + str(cumul) + ". " + str(i) + " " +  str(c) + " " + str(self.pprime[c]))

				prob.append(1)

				probc.append( prob )

			probV.append(probc)

		terms = [ ]

		for i in range(n_clauses):
			c = 0

			imp = []
			n = arity
			while n > 0:
				n0 = n - self.select_imp(probV[n][c], np.random.random())
				imp.append( n0 )

				n = n - n0
				if n > 0:
					if c < self.n_class - 2:
						c += 1
					else:
						imp.append(n)
						n = 0

			for n in range(len(imp), self.n_class):
				imp.append(0)

			var = [ ]
			for k in range(self.n_class):
				nk = imp[k]
				if (nk > 0):
					if (nk <= len(imp_var["vars_in_class"][ k ])): # !! nk should be lower or equal than the number of variables of importance imp[k]
						choice = np.random.choice(imp_var["vars_in_class"][ k ], nk, replace = False)
						var = var + choice.tolist()
					else:
						print ("error: number of variables for this sub-problem is larger than available of this importance\n")

			terms.append(var)

		return(terms)

	def select_imp(self, prob, x):
		i = 0
		while (prob[i] < x) and (i < len(prob)):
			i = i + 1
		return i

	def create_weigths(self, imp_var, terms):
		if (self.typeWeight == 0):
			w = [1]* len(terms)
		else :
			w = []
			for t in terms :
				v = [ ]
				for x in t:
					cl = imp_var["var_importance"][ x ]
					v.append(self.importance["degree"][cl])
				
				w.append( int(np.exp(np.mean(np.log(v)))) )
			
		return(w)


class PortfolioBuilder:
	def __init__(self, n):
		# problem dimension
		self.n = n

	def make(self):
		# portfollio from Chook generator / Tile Planting instances
		fc1 = FunctionClass(0, self.n, -5, [(-2, [0, 1]), (-2, [1, 2]), (1, [2, 3]), (-2, [0, 3]) ])
		fc2 = FunctionClass(1, self.n, -4, [(-2, [0, 1]), (-2, [1, 2]), (1, [2, 3]), (-1, [0, 3]) ])
		fc3 = FunctionClass(2, self.n, -3, [(-1, [0, 1]), (-1, [1, 2]), (1, [2, 3]), (-2, [0, 3]) ])
		fc4 = FunctionClass(3, self.n, -2, [(-1, [0, 1]), (-1, [1, 2]), (1, [2, 3]), (-1, [0, 3]) ])

		return [fc1, fc2, fc3, fc4]

class FunctionClass:
	def __init__(self, id, n, optimum, terms):
		# id of the function
		self.id = id
		# problem dimension
		self.n = n
		# value (fitness) of the optimum
		self.optimum = optimum
		# list of term of the function
		self.terms = terms

		self.arity = 0
		for t in terms:
			for x in t[1]:
				if self.arity < x:
					self.arity = x
		self.arity += 1

	def make(self, var):
		p = we.WalshExpansion(self.n)

		for t in self.terms:
			ids = []
			for x in t[1]:
				ids.append( var[x] )
			ids.sort()
			p.addTerm(t[0], tuple(ids))

		return p

