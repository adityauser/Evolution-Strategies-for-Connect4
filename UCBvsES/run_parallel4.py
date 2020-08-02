import os                                                                       
from multiprocessing import Pool


processes = (
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 8 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 8 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 16 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 16 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 32 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 32 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 64 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 64 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 128 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 128 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 8 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 8 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 16 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 16 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 32 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 32 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 64 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 64 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 128 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.1 --arms 10 --hidden 128 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 8 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 8 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 16 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 16 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 32 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 32 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 64 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 64 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 128 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 128 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v1",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 8 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 8 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 16 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 16 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 32 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 32 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 64 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 64 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 128 --noise --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2",
    "UCBvsES.py --mutation_mag 0.01 --arms 10 --hidden 128 --save_result --layers 3 --max_gen 200 --trials 100 --pop_size 150 --name resetEnv50_v2"
)

                                             
                                                  
def run_process(process):                                                             
    os.system('python {}'.format(process))                                       
                                                                                                                                                      
pool = Pool(processes=80)
pool.map(run_process, processes) 

