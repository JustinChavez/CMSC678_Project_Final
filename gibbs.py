import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

def gibbs_sampling(theta1_init, theta2_init, sd1, sd2, corr, itr):
    #initialize values
    
    theta1 = theta1_init
    theta2 = theta2_init
    samples = np.empty((itr,2))
    samples[0] = [theta1, theta2]
    #generate samples
    for i in range(itr - 1):
        #resample theta values
        theta1 = np.random.normal(corr*theta2, (1 - (corr*corr)))
        theta2 = np.random.normal(corr*theta1, (1 - (corr*corr)))
        samples[i + 1] = [theta1, theta2]
        
    return samples
        
