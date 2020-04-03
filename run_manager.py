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
    def __init__(self,original_filename,number_of_total_runs,number_of_cpus,path = os.getcwd()):
        self.number_of_total_runs = number_of_total_runs
        self.number_of_cpus = number_of_cpus
        self.path = path
        
        if original_filename[-3:] == '.py': original_filename = original_filename[:-3]
        self.original_filename = original_filename
        return
    
    def _delete_runned_files(self,files_names):
        current_path = os.getcwd()
        for x in files_names:
            os.remove( os.path.join(current_path,x) )    
        return
    
    def _make_running_files(self):
        files_names_list = [ '{0}({1}).py'.format(self.original_filename,i) for i in range(self.number_of_total_runs)]
        for i in range(self.number_of_total_runs):
            filename_plus_address = os.path.join(self.path , self.original_filename)
            with open(filename_plus_address + '.py', "r") as f:
                old = f.read() # read everything in the file
            with open(filename_plus_address + "({0}).py".format(i), "w") as f:
                f.seek(0) # rewind
                f.write("version = 'Result {0}_{1}'\n".format(self.original_filename,i) + old) # write the new line before
                f.close()
        return files_names_list
    
    def _prompt_os_to_run(self,file_name):
        proc = os.getpid()
        print('{} runned by process id: {}'.format(
            file_name, proc))
        
        # current_path = os.getcwd()
        # genres_path = os.path.join(current_path,'genres_holder')
        # os.system('cd {}'.format(genres_path))
        files_str = '{} &'.format(file_name)
        os.system('python ' + files_str)
        time.sleep(5)
        return
    
    def initialize_running(self):
        self.files_names = self._make_running_files()
        pool = Pool(processes = self.number_of_cpus)
        pool.map(self._prompt_os_to_run,self.files_names)
        pool.close()
        pool.join()
        # self._delete_runned_files(files_names)
        return

    pass

if __name__ == '__main__':
    genres_name_list = ['Homans_1a.py','Homans_1b.py']
    current_path = os.getcwd()
    genres_path = os.path.join(current_path,'genres_holder')
    for genre in tqdm.tqdm(genres_name_list):
        first = Genre_execution(genre, 4, 2)
        first.initialize_running()
        print('done')


        