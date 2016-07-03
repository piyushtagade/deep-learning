import numpy as np 
import scipy as sc 
import math

# ---------------------------------------------------------
#                    Class for MCMC
#  Creates an MCMC object and define methods for sampling 
# ---------------------------------------------------------

class MarkovChainMonteCarlo:
     ''' mcmc sampling '''
     def __init__(self, number_samples, potential, potential_grad, epsilon=None, number_steps=None):
         ''' Initializing the mcmc object ''' 
         self.number_samples = number_samples # Number of samples
         self.potential = potential 
         self.potential_grad = potential_grad # Gradient of the potential 
         if epsilon is None:
             self.epsilon = 0.3 
         else: 
             self.epsilon = epsilon
         
         if number_steps is None:
             self.number_steps = 10 
         else: 
             self.number_steps = number_steps

# ---------------------------------------------------------
#  Hamiltonian Monte Carlo for sampling 
# ---------------------------------------------------------

# Options for HMC
     def options(self, epsilon, number_steps): 
         ''' Setting options for hmc ''' 
         self.epsilon = epsilon
         self.number_steps = number_steps    

# --------------------------------------------------------------
# Leap-frog method for numerical integration of the Hamiltonian 
# --------------------------------------------------------------

     def leap_frog(self, position, momentum): 
         ''' Leap frog method for numerical integration ''' 
         epsilon = self.epsilon 
         numsteps = self.number_steps
         gradU = self.potential_grad
# -------------------------------------------------------------
# Initializing 
# -------------------------------------------------------------
         q = position
         p = momentum     
# --------------------------------------------------------------
# Leap-frog 
# --------------------------------------------------------------
         for steps in range(0, numsteps): 
             p = p - (epsilon/2.0)*gradU(q) 
             q = q + epsilon*p 
             p = p - (epsilon/2.0)*gradU(q) 
  
         position = q; momentum = p
         return position, momentum


# --------------------------------------------------------------
# MCMC Sampling using Hammiltonian Monte Carlo 
# --------------------------------------------------------------

     def sample(self, q0):
         ''' Samples using HMC ''' 
         samples = [q0]; 
         numsamps = self.number_samples
         potential = self.potential 
         q = q0; 
         cov = np.eye(q.shape[0])  
         for samp in range(0, numsamps):
             p = np.random.multivariate_normal(np.zeros(q.shape[0]), cov, 1)
             #print(p, np.dot(np.dot(p.T, cov), p))
             #input('Press ENTER to continue') 
             h = potential(q) + np.dot(np.dot(p.T, cov), p)/2.0 
             qn, pn = self.leap_frog(q, p)
             pn = -pn
             hn = potential(qn) + np.dot(np.dot(pn.T, cov), pn)/2.0
             acc = math.exp(-hn + h)
             urand = np.random.random(1) 
             if (urand < acc):
                 q = qn
             else:
                 q = q 
             #print(urand, acc, q, qn)
             #input('Press ENTER to continue') 
             samples.append(q) 
         return np.array(samples) 

             # Potential energy     
