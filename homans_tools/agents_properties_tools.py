# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:10:32 2019

@author: vaio
"""
import matplotlib.pyplot as plt
import numpy as np

class arrays_glossary():
    def array(self,what_array):
        ref = {}
        
        if what_array == 'degree':
            ref[what_array] = np.zeros(self.N)
            for n in range(self.N):   
                ref[what_array][n] = self.G.degree(n) if n in self.G.nodes() else 0
            return ref[what_array]

        if what_array == 'neighbor':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].neighbor
            return ref[what_array]
        
        if what_array == 'value':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    if self.a_matrix[i].neighbor[j] < self.memory_size:
                        where = self.a_matrix[i].neighbor[j]-1 #last value in memory
                    else:
                        where = self.memory_size-1
                    ref[what_array][i,j] = self.a_matrix[i].value[j, where ]
            return ref[what_array]
            
        if what_array == 'time':
            ref[what_array] = np.zeros((self.N,self.N,self.memory_size))
            for i in np.arange(self.N):
                for j in np.arange(self.N):
                    if self.a_matrix[i].neighbor[j] != 0:
                        ref[what_array][i,j] = self.a_matrix[i].time[j]
                        #ref[what_array][i,j] = self.a_matrix[i].value[j]
                    else:
                        ref[what_array][i,j] = -1
            return ref[what_array]
        
        if what_array == 'probability':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    ref[what_array][i,j] = self.a_matrix[i].active_neighbor[j]
            return ref[what_array]

        if what_array == 'utility':
            ref[what_array] = np.zeros((self.N,self.N))
            for i in np.arange(self.N):
                for j in self.a_matrix[i].active_neighbor.keys():
                    ref[what_array][i,j] = self.a_matrix[i].active_neighbor[j] * self.a_matrix[j].active_neighbor[i]
            return ref[what_array]
        
        if what_array == 'others_feeling':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array] += self.a_matrix[i].feeling[:]
            return ref[what_array]/np.sum(ref[what_array])

                    
        if what_array == 'worth_ratio':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].worth_ratio
            return ref[what_array]
        
        if what_array == 'situation':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].situation
            return ref[what_array]
        
        if what_array == 'asset':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].asset
            return ref[what_array]
        
        if what_array == 'money':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].money
            return ref[what_array]
        
        if what_array == 'approval':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = self.a_matrix[i].approval
            return ref[what_array]
        
        if what_array == 'active_neighbor':
            ref[what_array] = np.zeros(self.N)
            for i in np.arange(self.N):
                ref[what_array][i] = len(self.a_matrix[i].active_neighbor)
            return ref[what_array]
        
        if what_array == 'community':
            community_dict = self.best_parts
            community_array = np.zeros(self.N) -1 #minus one means isolated
            for i in range(self.N):
                if i in community_dict.keys():
                    community_array[i] = community_dict[i]
            ref[what_array] = np.copy(community_array)
            return ref[what_array]
        
        if what_array == 'self_value':
            ref[what_array] = np.sum(self.array('value'),axis = 1)
            return ref[what_array]
        
        if what_array == 'value_to_others':
            ref[what_array] = np.sum(self.array('value'),axis = 0)
            return ref[what_array]
    pass
