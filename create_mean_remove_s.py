import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import datetime
from numba import njit
"""
e_0.append(e[0][0])

e_3.append(e[0][1])

e_6.append(s[0][0])

e_9.append(tau[0][0])

e_12.append(taud[0][0])

e_15.append(y[0][0])

e_18.append(taus0[0][0])

e_21.append(D[0][0])

e_24.append(beta.T @ omega)
e_25.append(np.linalg.norm(s))
e_26.append(np.linalg.norm(e[:,0:1]))
"""

def main(dir_base_0,alpha_lambda_0,alpha_sab1):

    alpha_sa1_0 = alpha_sab1
    alpha_sb1_0 = alpha_sab1
    alpha_sm1_0 = 2.0
    alpha_sm2_0 = 0.6

    # alpha_lambda_1 = 0.0
    # alpha_sa1_1 = 1.0
    # alpha_sb1_1 = 1.0
    # alpha_sm1_1 = 2.0
    # alpha_sm2_1 = 0.6
    # dir_base_1 = "./data/no/"

    T = 1000
    step = 0.0001
    end = 100

    end_plt = 100
    start_plt = 0

    # t_data = np.loadtxt(f"./data/step{step}_t{end}.csv",delimiter = ",")

    n = 100

    m = int(end/step/10)

    e0 = np.zeros((n,m))
    e1 = np.zeros((n,m))
    e2 = np.zeros((n,m))

    norm = np.zeros((n,m))

    e_mean = np.zeros((7,m))

    list = [1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 24, 25, 27, 31, 33, 35, 36, 37, 38, 39, 40, 42, 46, 47, 48, 50, 52, 53, 55, 56, 58, 59, 60, 64, 66, 67, 68, 70, 72, 73, 76, 77, 80, 81, 82, 83, 86, 87, 89, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 107, 108, 109, 112, 113, 114, 116, 117, 119, 121, 123, 125, 126, 127, 128, 129, 133, 134, 136, 137, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149, 151, 152, 158]

    for i in tqdm(range(n)):
        
        e_all_p = np.load(dir_base_0 + f"s{list[i]}_m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_e_all.npy")
        
        e0[i] = e_all_p[0]
        e1[i] = e_all_p[1]
        e2[i] = e_all_p[2]
        norm[i] = e_all_p[26]

    e0_mean = np.mean(e0,axis=0)
    e0_abs = np.abs(e0)
    e0_abs_mean = np.mean(e0_abs,axis=0)

    e1_mean = np.mean(e1,axis=0)
    e1_abs = np.abs(e1)
    e1_abs_mean = np.mean(e1_abs,axis=0)

    e2_mean = np.mean(e2,axis=0)
    e2_abs = np.abs(e2)
    e2_abs_mean = np.mean(e2_abs,axis=0)

    norm_mean = np.mean(norm,axis=0)

    e_mean[0] = e0_mean
    e_mean[1] = e1_mean
    e_mean[2] = e2_mean
    e_mean[3] = e0_abs_mean
    e_mean[4] = e1_abs_mean
    e_mean[5] = e2_abs_mean
    e_mean[6] = norm_mean

    os.makedirs(dir_base_0 + "mean/", exist_ok=True)

    np.save(dir_base_0 + f"mean/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_mean_remove.npy",e_mean)
    np.save(dir_base_0 + f"mean/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_norm_remove.npy",norm)



if __name__ == '__main__':
    
    start = time.perf_counter()
    
    print(datetime.datetime.now())
    
    use_cpu = 20
    
    print(f"use cpu core {use_cpu}/{cpu_count()}")

    init = []

    # list_dir_base_0 = ["./data/b_no/","./data/s_no/","./data/s_bzd/","./data/s_bzdm/"]
    list_dir_base_0 = ["./data/s_no/","./data/s_bzd/","./data/s_bzdm/"]

    list_alpha_lambda = [0.0,0.3,0.5]
    
    list_alpha_sab1 = [1.0,2.0,3.0]
    
    for i in list_dir_base_0:
        
        for j in list_alpha_lambda:
            
            for k in list_alpha_sab1:
                
                init.append((i,j,k))
        

    with Pool(use_cpu) as p:
        
        r = p.starmap(func=main,iterable=init)
        
    print(time.perf_counter() - start)