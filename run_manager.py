# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:32:43 2020

@author: mohsen,taha
"""
import os
import time
from multiprocessing import Pool, Process
import tqdm
"""
for running all the number of desired run we split the total number into
number of cycles. each cycle has num_run_in_cycle in its body.
"""

class Genre_execution():
    def __init__(self,original_filename,number_of_total_runs,number_of_cpus,total_time, path = os.getcwd(), versions = []):
        self.number_of_total_runs = number_of_total_runs
        self.number_of_cpus = number_of_cpus
        self.path = path
        
        if original_filename[-3:] == '.py': original_filename = original_filename[:-3]
        self.original_filename = original_filename
        self.versions_names = versions
        
        self.total_time = total_time
        return
    
    def _delete_runned_files(self):
        # current_path = os.getcwd()
        for x in self.files_names:
            os.remove( os.path.join(self.path,x) )    
        return
    
    def _make_homans_running_files(self):
        files_names_list = [ '{0}({1}).py'.format(self.original_filename,i) for i in range(self.number_of_total_runs)]
        for i in range(self.number_of_total_runs):
            filename_plus_address = os.path.join(self.path , self.original_filename)
            with open(filename_plus_address + '.py', "r") as f:
                old = f.read() # read everything in the file
            with open(filename_plus_address + "({0}).py".format(i), "w") as f:
                f.seek(0) # rewind
                version_name = 'Result {0}_{1}'.format(self.original_filename,i)
                self.versions_names.append(version_name)
                f.write("T = {}".format(self.total_time)+"\n"+
                        "version = " + "'{}'".format(version_name)+
                        "\n" + old) # write the new line before
                f.close()
        self.files_names = files_names_list
        return 
    
    def _make_results_analysis_running_files(self):
        files_names_list = [ '{0}({1}).py'.format(self.original_filename,i) for i in range(self.number_of_total_runs)]
        
        for i in range(self.number_of_total_runs):
            filename_plus_address = os.path.join(self.path , self.original_filename)
            with open(filename_plus_address + '.py', "r") as f:
                old = f.read() # read everything in the file
            with open(filename_plus_address + "({0}).py".format(i), "w") as f:
                f.seek(0) # rewind
                version_name = self.versions_names[i]
                f.write("T = {}".format(self.total_time)+"\n"+
                        "version = " + "'{}'".format(version_name)+
                        "\n" + old) # write the new line before
                f.close()
        
        self.files_names = files_names_list        
        return
    
    def _prompt_os_to_run(self,file_name):
        files_str = os.path.join(self.path,'{} &'.format(file_name))
        os.system('python ' + files_str)
        time.sleep(5)
        return
    
    def initialize_running(self):

        if self.original_filename[:6] == 'Homans':
            self._make_homans_running_files()
        elif self.original_filename[:7] == 'Results':
            self._make_results_analysis_running_files()
            
        pool = Pool(processes = self.number_of_cpus)
        for _ in tqdm.tqdm(pool.map(self._prompt_os_to_run,self.files_names), total=len(self.files_names)):
            pass
        pool.close()
        pool.join()
        self._delete_runned_files()
        return

    pass

if __name__ == '__main__':
    """Homans files"""
    total_running_steps = 5000
    cpus_at_hand = 5
    each_file_run_times = 10
    genres_name_list = ['Homans_1_a.py','Homans_1_b.py','Homans_2_a.py','Homans_2_b.py',
                        'Homans_3_a.py','Homans_3_b.py','Homans_3_c.py']
    total_files = each_file_run_times * len(genres_name_list)
    genres_versions_names = []
    
    control_file = open('check_status.txt','w')
    
    current_path = os.getcwd()
    genres_path = os.path.join(current_path,'genres_holder')
    for i,genre in enumerate(tqdm.tqdm(genres_name_list)):
        genre_homans_exec = Genre_execution(genre, each_file_run_times , cpus_at_hand,
                                            total_running_steps, versions = [], path = genres_path)
        genre_homans_exec.initialize_running()
        genres_versions_names.extend(genre_homans_exec.versions_names)
        control_file.write('{} completed! \n'.format(genre))
    
        
    """"Analysis part"""
    genre_results_exec = Genre_execution('Results_analysis_Homans.py', len(genres_versions_names), cpus_at_hand,
                                          total_running_steps, path = genres_path,
                                          versions = genres_versions_names)
    genre_results_exec.initialize_running()
    # for start in tqdm.tqdm( range(0,len(genres_versions_names),3) ):
    #     end = (start + 3) if start < ( len(genres_versions_names) - 3 ) else len(genres_versions_names)
    #     genre_results_exec = Genre_execution('Results_analysis_Homans.py', end - start, end - start,
    #                                       total_running_steps, path = genres_path,
    #                                       versions = genres_versions_names[start:end])
    #     genre_results_exec.initialize_running()
    #     control_file.write('{} completed! \n'.format(genres_versions_names[start:end]))
        
    control_file.write('all results generation completed! \n')
    control_file.write('Program Jobs are Finished! \n')

        