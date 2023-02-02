import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import datetime
from numba import njit

def main(a,b,c):
    
    print(f"{a},{b},{c}")

if __name__ == '__main__':
    
    start = time.perf_counter()
    
    print(datetime.datetime.now())
    
    use_cpu = 20
    
    print(f"use cpu core {use_cpu}/{cpu_count()}")

    init = []

    list_m = [0.0,0.3,0.5]
    
    list_ab = [1.0,2.0,3.0]

    for i in range(4):
        
        for j in list_ab:
            
            for k in list_m:
                
                init.append((i,j,k))
        

    with Pool(use_cpu) as p:
        
        r = p.starmap(func=main,iterable=init)
        
    print(time.perf_counter() - start)