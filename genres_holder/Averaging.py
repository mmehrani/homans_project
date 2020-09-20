"""
Created on Tue Apr 21 18:32:40 2020

@author: Taha, Mohsen
"""
import numpy as np
import os
import pandas as pd

class Version_result():
    def __init__(self,run_name,total_time):
        self.run_name = run_name
        self.total_time = total_time
        pass
    
    def average_on_ensumbles(self):
        """
        computes communities parameters for the results of ensumbles for
        a defined version

        """
        run_array = range(35) #assign the maximum number of runs
        file_to_look = ['Communities.py']
        initial_path = os.path.join(os.getcwd(), 'runned_files','N100_T{}'.format(self.total_time))        
        modul, modul_r, cover, cover_r, num_com = [], [], [], [], []
        aspl, aspl_r, cc, cc_r, sigma, omega, edges, nodes = [], [], [], [], [], [], [], []
        a_asset, a_money, a_appr, a_sit, a_wr = [], [], [], [], []
        """Creating Arrays"""
        for i in run_array:
            path = os.path.join(initial_path, self.run_name+ str(i), '0_{}'.format(self.total_time))
            # path = initial_path + '\\' + self.run_name + str(i) + '\\0_5000\\'
            for file_to_look in ['Communities.txt','Topological Charateristics.txt','Assortativity.txt']:
                try:
                    with open( os.path.join(path , file_to_look) ,'r') as file:
                        lines = file.readlines()
                        if file_to_look == 'Communities.txt':
                            modul.append(float(lines[1][:-1]))
                            cover.append(float(lines[3][:-1]))
                            modul_r.append(float(lines[5][:-1]))
                            cover_r.append(float(lines[7][:-1]))
                            num_com.append(float(lines[9][:-1]))
                        if file_to_look == 'Topological Charateristics.txt':
                            aspl.append(float(lines[2][:-1]))
                            cc.append(float(lines[4][:-1]))
                            sigma.append(float(lines[6][:-1]))
                            omega.append(float(lines[8][:-1]))
                            aspl_r.append(float(lines[11][22:-1]))
                            cc_r.append(float(lines[12][23:-1]))
                            where = lines[0].find('w')
                            edges.append(int(lines[0][where+5:where+9]))
                            nodes.append(int(lines[0][where-4:where]))
                        if file_to_look == 'Assortativity.txt':
                            a_asset.append(float(lines[1][:-1]))
                            a_money.append(float(lines[4][:-1]))
                            a_appr.append(float(lines[7][:-1]))
                            a_sit.append(float(lines[10][:-1]))
                            a_wr.append(float(lines[13][:-1]))
                except: pass
        
        dic = {'modul':modul, 'modul_r':modul_r, 'cover':cover, 'cover_r':cover_r, 'num_com':num_com,
               'aspl':aspl,'aspl_r':aspl_r, 'cc':cc, 'cc_r':cc_r, 'sigma':sigma, 'omega':omega, 'nodes':nodes,
               'edges':edges, 'a_asset':a_asset, 'a_money':a_money, 'a_appr':a_appr, 'a_sit':a_sit, 'a_wr':a_wr}

        average = {}
        error = {}
        relative_error = {}
        normalized_to_random = {}
        normalized_error = {}

        for dic_item in dic:
            average[dic_item] = np.average(np.array(dic[dic_item]))
            error[dic_item + '_err'] = np.sqrt(np.var(np.array(dic[dic_item]),ddof=1)) / np.sqrt(len(dic[dic_item]))
            relative_error[dic_item + '_rel_err'] = np.abs(error[dic_item + '_err'] / average[dic_item])
        for dic_item in ['modul','cover','aspl','cc']:    
            normalized_to_random[dic_item + '_norm'] = average[dic_item] / average[dic_item + '_r']
            normalized_error[dic_item + '_norm_err'] = np.abs(normalized_to_random[dic_item+'_norm']) * np.sqrt( (error[dic_item+'_err']/average[dic_item])**2 + (error[dic_item+'_r'+'_err']/average[dic_item+'_r'])**2 )
            # normalized_error[dic_item + '_norm_err'] = np.abs((error[dic_item+'_err']*average[dic_item+'_r'] - error[dic_item+'_r'+'_err']*average[dic_item]) / average[dic_item+'_r']**2)
        average['sigma_f'] = (average['aspl_r']*average['cc']) / (average['cc_r']*average['aspl'])
        error['sigma_f_err'] = np.abs(average['sigma_f']) * np.sqrt(sum((error[item+'_err']/average[item])**2 for item in ['cc','cc_r','aspl','aspl_r']))
        relative_error['sigma_f'+'_rel_err'] = np.abs(error['sigma_f'+'_err'] / average['sigma_f'])
        
        self.num_successful_runs = len(modul)
        average.update(error)
        average.update(relative_error)
        average.update(normalized_to_random)
        average.update(normalized_error)
        self.avg_std = average
        return
        
if __name__ == '__main__':
    total_running_steps = 5000
    
    genres_path = os.getcwd()
    version_folder_directory = os.path.join(genres_path,'runned_files','N100_T{}'.format(total_running_steps))
    
    genres_versions_names = []
    for entry_name in os.listdir(version_folder_directory):
        entry_path = os.path.join(version_folder_directory, entry_name)
        if os.path.isdir(entry_path):
            if (not entry_name[-2].isnumeric()) and (entry_name[:-1] not in genres_versions_names):
                genres_versions_names.append(entry_name[:-1])
            if entry_name[-2].isnumeric() and (entry_name[:-2] not in genres_versions_names):
                genres_versions_names.append(entry_name[:-2])
    
    columns_names = []
    parameters = ['modul', 'modul_r','cover','cover_r','aspl','aspl_r','cc','cc_r','sigma',
                  'sigma_f','omega','a_asset','a_money','a_appr','a_sit','a_wr','num_com','nodes','edges']
    for x in parameters: 
        columns_names.extend([x,x + '_err', x + '_rel_err'])
        if x in ['modul_r','cover_r','aspl_r','cc_r']:
            columns_names.extend([x[:-2] + '_norm',x[:-2] + '_norm_err'])
    df_all_genres_results = pd.DataFrame(columns = columns_names)
    
    for run_name in genres_versions_names:
        genre_result = Version_result(run_name,total_running_steps)
        genre_result.average_on_ensumbles()
        
        new_data = genre_result.avg_std
        new_data['runs'] = genre_result.num_successful_runs
        
        new_row = pd.Series(data=new_data, name=run_name)
        df_all_genres_results = df_all_genres_results.append(new_row, ignore_index=False)
    df_all_genres_results.to_csv('Final_Data.csv')
    