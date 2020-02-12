# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 08:31:14 2020

@author: mohsen
"""

"""Parameters"""#XXX
import numpy as np

N = 50
T = 500
similarity = 0.05                   #how much this should be?
memory_size = 10                    #contains the last memory_size number of transaction times
transaction_percentage = 0.1        #percent of amount of money the first agent proposes from his asset 
num_of_tries = 20                   #in function explore()
threshold_percentage =np.full(N,1)  #the maximum amount which the agent is willing to give
normalization_factor = 1            #used in transaction(). what should be?
prob0_magnify_factor = 0.3          #this is in probability() for changing value so that it can take advantage of arctan
prob1_magnify_factor = 2
prob2_magnify_factor = 1
alpha = 1                           #in short-term effect of the frequency of transaction
beta = 0.3                          #in long-term effect of the frequency of transaction
param = 2                           #a normalizing factor in assigning the acceptance probability. It normalizes difference of money of both sides
lamda = 0                           # how much one agent relies on his last worth_ratio and how much relies on current transaction's worth_ratio
sampling_time = 500
saving_time_step = 500
initial_for_trans_time = 0
trans_saving_interval = 500
version = '1_basic_run'