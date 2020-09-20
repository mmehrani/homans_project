"""
Created on Sun May  3 16:20:22 2020

@author: Taha Enayat
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

class Version_result():
    def __init__(self,run_name,total_time):
        self.run_name = run_name
        self.total_time = total_time
        pass

    def plot_general(self,path,array,title='',second_array=None,indicator=True,**kwargs):
        """ 
        Used in graph_related function
        """
        plt.figure()
        if indicator:
            plt.plot(array,label = 'Homans')
        else:
            label = kwargs.get('label',None)
            if label != None:
                for i in np.arange(len(array)):
                    plt.plot(array[i],label=label[i])
                plt.legend()
            else: 
                for i in np.arange(len(array)):
                    plt.plot(array[i])
        if np.any(second_array) != None:
            plt.plot(second_array, label = 'random pair')
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(path , title))
        plt.close()
        return
    
    def errorbar_general(self,path,array,array_err,title='',second_array=None,second_array_err=None,indicator=True,**kwargs):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.errorbar(range(len(array)), array, yerr=array_err, fmt='o', label = 'Homans')
        if np.any(second_array) != None: ax.errorbar(range(len(second_array)), second_array, yerr=second_array_err, fmt='o',label = 'random pair')
        
        plt.legend()
        plt.title('Case ' + title)
        plt.savefig(os.path.join(path , title  + '.pdf'))
        plt.close()        
        return
    
    def average_on_ensumbles(self):
        """
        computes communities parameters for the results of ensumbles for
        a defined version

        """
        run_array = range(35) #assign the maximum number of runs
        gr = 30
        initial_path = os.path.join(os.getcwd(), 'runned_files','N100_T{}'.format(self.total_time))
        dic = {'modul':[], 'modul_r':[], 'cover':[], 'cover_r':[], 'aspl':[],'aspl_r':[], 'cc':[],
               'cc_r':[], 'omega':[],'a_asset':[], 'a_money':[], 'a_appr':[], 'a_wr':[]}

        """Creating Arrays"""
        counter = 0
        for i in run_array:
            path = os.path.join( initial_path, self.run_name + str(i), '0_{}'.format(self.total_time), 'graph_related')
            # path = initial_path + '\\' + self.run_name + str(i) + '\\0_5000' + '\\graph_related\\'
            if os.path.isdir(path): 
                for x in dic.keys(): dic[x].append([])
                counter += 1
            for fpoint in np.arange(gr)+1:
                for file_to_look in ['Communities.txt','Topological Charateristics.txt','Assortativity.txt']:
                    try:
                        with open( os.path.join(path, str(fpoint) + ', ' + file_to_look),'r' ) as file:
                        # with open( os.path.join(path, str(fpoint) + ', ' + file_to_look),'r' ) as file:
                            lines = file.readlines()
                            if file_to_look == 'Communities.txt':
                                dic['modul'][-1].append(float(lines[1][:-1]))
                                dic['cover'][-1].append(float(lines[3][:-1]))
                                dic['modul_r'][-1].append(float(lines[5][:-1]))
                                dic['cover_r'][-1].append(float(lines[7][:-1]))
                            if file_to_look == 'Topological Charateristics.txt':
                                dic['aspl'][-1].append(float(lines[2][:-1]))
                                dic['cc'][-1].append(float(lines[4][:-1]))
                                # dic['sigma'][-1].append(float(lines[6][:-1]))
                                dic['omega'][-1].append(float(lines[8][:-1]))
                                dic['aspl_r'][-1].append(float(lines[11][22:-1]))
                                dic['cc_r'][-1].append(float(lines[12][23:-1]))
                            if file_to_look == 'Assortativity.txt':
                                dic['a_asset'][-1].append(float(lines[1][:-1]))
                                dic['a_money'][-1].append(float(lines[4][:-1]))
                                dic['a_appr'][-1].append(float(lines[7][:-1]))
                                dic['a_wr'][-1].append(float(lines[13][:-1]))
                    except: pass
        # print(dic['modul'])
        # print(dic['modul_r'])
        # print(dic)
        average = {}
        error = {}
        for dic_item in dic:
            average[dic_item] = np.average(np.array(dic[dic_item]),axis=0)
            error[dic_item + '_err'] = np.sqrt(np.var(np.array(dic[dic_item]),axis=0,ddof=1)) / np.sqrt(len(dic[dic_item]))
        
        average['sigma'] = (average['aspl_r']*average['cc']) / (average['cc_r']*average['aspl'])
        temp_sum = np.zeros(np.size(average['sigma']))
        for item in ['cc','cc_r','aspl','aspl_r']: temp_sum += (error[item+'_err']/average[item])**2
        error['sigma_err'] = np.abs(average['sigma']) * np.sqrt(temp_sum)

        # print(average['sigma'])
        # print(np.size(error['sigma_err']))
        
        self.num_successful_runs = len(dic['modul'])
        average.update(error)
        self.avg_std = average
        return average, error
        
if __name__ == '__main__':
    total_running_steps = 5000
    
    genres_path = os.getcwd()
    version_folder_directory = os.path.join(genres_path,'runned_files','N100_T{}'.format(total_running_steps))
    
    genres_versions_names = []
    # for entry_name in os.listdir(version_folder_directory):
    #     entry_path = os.path.join(version_folder_directory, entry_name)
    #     if os.path.isdir(entry_path):
    #         if (not entry_name[-2].isnumeric()) and (entry_name[:-1] not in genres_versions_names):
    #             genres_versions_names.append(entry_name[:-1])
    #         if entry_name[-2].isnumeric() and (entry_name[:-2] not in genres_versions_names):
    #             genres_versions_names.append(entry_name[:-2])
    genres_versions_names = ['Result_Homans_1_a_','Result_Homans_2_a_',
                             'Result_Homans_3_a_','Result_Homans_3_b_']
    average = {}
    error = {}
    successful_runs = []
    for i,run_name in enumerate(genres_versions_names):
        genre_result = Version_result(run_name,total_running_steps)
        average[i], error[i] = genre_result.average_on_ensumbles()
        successful_runs.append(genre_result.num_successful_runs)
        
    initial_path = os.path.join(os.getcwd(), 'runned_files')
    try: os.mkdir( os.path.join(initial_path,'Ensembles') )
    except: pass
    path = os.path.join(initial_path, 'Ensembles')
    # path = initial_path + '\\Ensembles\\'
    for i,run_name in enumerate(genres_versions_names):
        genre_result.plot_general(path,average[i]['modul'],second_array=average[i]['modul_r'],title=run_name[14:-1]+' Modularity ')
        genre_result.plot_general(path,average[i]['cover'],second_array=average[i]['cover_r'],title=run_name[14:-1]+' Coverage ')
        genre_result.plot_general(path,average[i]['sigma'],title=run_name[14:-1]+' Smallworldness Sigma ')
        genre_result.plot_general(path,average[i]['omega'],title=run_name[14:-1]+' Smallworldness Omega ')
        genre_result.plot_general(path,average[i]['cc'],second_array=average[i]['cc_r'],title=run_name[14:-1]+' Clustering Coefficient ')
        genre_result.plot_general(path,average[i]['aspl'],second_array=average[i]['aspl_r'],title=run_name[14:-1]+' Shortest Path Length ')
        genre_result.plot_general(path,np.array(average[i]['cc'])/np.array(average[i]['cc_r']),title=run_name[14:-1]+' Clustering Coefficient Normalized ')
        genre_result.plot_general(path,np.array(average[i]['aspl'])/np.array(average[i]['aspl_r']),title=run_name[14:-1]+' Shortest Path Length Normalized ')
        
        

    for i,run_name in enumerate(genres_versions_names):
        genre_result.plot_general(path,error[i]['modul'+'_err'],second_array=error[i]['modul_r'+'_err'],title=run_name[14:-1]+' error'+' Modularity ')
        genre_result.plot_general(path,error[i]['cover'+'_err'],second_array=error[i]['cover_r'+'_err'],title=run_name[14:-1]+' error'+' Coverage ')
        genre_result.plot_general(path,error[i]['sigma'+'_err'],title=run_name[14:-1]+' error'+' Smallworldness Sigma ')
        genre_result.plot_general(path,error[i]['omega'+'_err'],title=run_name[14:-1]+' error'+' Smallworldness Omega ')
        genre_result.plot_general(path,error[i]['cc'+'_err'],second_array=error[i]['cc_r'+'_err'],title=run_name[14:-1]+' error'+' Clustering Coefficient ')
        genre_result.plot_general(path,error[i]['aspl'+'_err'],second_array=error[i]['aspl_r'+'_err'],title=run_name[14:-1]+' error'+' Shortest Path Length ')

    for i,run_name in enumerate(genres_versions_names):
        genre_result.errorbar_general(path,average[i]['modul'],error[i]['modul'+'_err'],second_array=average[i]['modul_r'],second_array_err=error[i]['modul_r'+'_err'],title=run_name[14:-1]+' Modularity ')
        genre_result.errorbar_general(path,average[i]['cover'],error[i]['cover'+'_err'],second_array=average[i]['cover_r'],second_array_err=error[i]['cover_r'+'_err'],title=run_name[14:-1]+' Coverage ')
        genre_result.errorbar_general(path,average[i]['sigma'],error[i]['sigma'+'_err'],title=run_name[14:-1]+' Smallworldness Sigma ')
        genre_result.errorbar_general(path,average[i]['omega'],error[i]['omega'+'_err'],title=run_name[14:-1]+' Smallworldness Omega ')
        genre_result.errorbar_general(path,average[i]['cc'],error[i]['cc'+'_err'],second_array=average[i]['cc_r'],second_array_err=error[i]['cc_r'+'_err'],title=run_name[14:-1]+' Clustering Coefficient ')
        genre_result.errorbar_general(path,average[i]['aspl'],error[i]['aspl'+'_err'],second_array=average[i]['aspl_r'],second_array_err=error[i]['aspl_r'+'_err'],title=run_name[14:-1]+' Shortest Path Length ')
        # genre_result.errorbar_general(path,np.array(average[i]['cc'])/np.array(average[i]['cc_r']),title=run_name[14:-1]+' Clustering Coefficient Normalized ')
        # genre_result.errorbar_general(path,np.array(average[i]['aspl'])/np.array(average[i]['aspl_r']),title=run_name[14:-1]+' Shortest Path Length Normalized ')
    
    with open(path + 'successful_runs.txt','w') as run:
        for i,run_name in enumerate(genres_versions_names):
            run.write(run_name + ': ')
            run.write(str(successful_runs[i]))
            run.write('\n')
            
            
            
            