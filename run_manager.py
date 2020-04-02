# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:32:43 2020

@author: mohsen,taha
"""
import os
import time
import shutil
from multiprocessing import Pool, Process

"""
for running all the number of desired run we split the total number into
number of cycles. each cycle has num_run_in_cycle in its body.
"""
def check_parallel_runs():    
    return

def make_control_file():
    return

def make_running_files(num_run_total):
    files_names_list = [ 'Homans({}).py'.format(i) for i in range(num_run_total)]
    for i in range(num_run_total):
        with open("Homans.py", "r") as f:
            # lines = f.readlines()
            old = f.read() # read everything in the file
        with open("Homans({}).py".format(i), "w") as f:
            # old = f.read() # read everything in the file
            f.seek(0) # rewind
            f.write("version = 'Result #_alphabet_{}'\n".format(i) + old) # write the new line before
            f.close()
    return files_names_list

def initialize_running(files_names):
    
    proc = os.getpid()
    print('{} runned by process id: {}'.format(
        files_names, proc))
    files_str = ' {} &'.format(files_names)
    os.system('python' + files_str)
    time.sleep(5)
    return

if __name__ == '__main__':
    num_run_total = 3
    num_run_in_cycle = 3
    files_names = make_running_files(num_run_total)
    
    pool = Pool(processes = num_run_in_cycle)
    pool.map(initialize_running,files_names)
    pool.join()
    # procs = []
    # for index, file in enumerate(files_names):
    #     proc = Process(target=initialize_running, args=(file,))
    #     procs.append(proc)
    #     proc.start()

    # for proc in procs:
    #     if proc.is_alive(): proc.join()