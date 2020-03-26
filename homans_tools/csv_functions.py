# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:52:32 2020

@author: mohsen
"""

import csv
import numpy as np

def make_agents_datatable(analyse_object):
    """
    Parameters
    ----------
    analyse_object : Analyse object of Analysis_Tools_Homans.py
        the analyse object created in the Results_analysis_Homans.py .

    Returns
    -------
    agents_data.
        table of data with each row an agent and each colummn a property.
    """
    header_list = ['money','asset','approval','worth_ratio',
                 'situation','others_feeling','degree','community',
                 'self_value','value_to_others']
    
    agents_data = np.zeros( (len(header_list),analyse_object.N) )
    for index,prop in enumerate(header_list):
        agents_data[index,:] =  analyse_object.array(prop) 
    
    return header_list,agents_data.T

def data_to_csv(analyse_object):
    header,agents_data = make_agents_datatable(analyse_object)
    file = open(analyse_object.path + 'all_agents_properties.csv', 'w')

    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in agents_data:
            writer.writerow(row)
    return