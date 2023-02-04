import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import datetime
from numba import njit
from matplotlib import pyplot as plt

def main(dir_base_0,alpha_lambda_0,alpha_sab1):

    alpha_sa1_0 = alpha_sab1
    alpha_sb1_0 = alpha_sab1
    alpha_sm1_0 = 2.0
    alpha_sm2_0 = 0.6

    T = 1000
    step = 0.0001
    end = 100

    end_plt = 100
    start_plt = 0

    t_data = np.loadtxt(f"./data/step{step}_t{end}.csv",delimiter = ",")

    e_all_p = np.load(dir_base_0 + f"mean/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_norm_200.npy")
    # e_all_c = np.load(dir_base_1 + f"mean/m{alpha_lambda_1}_a{alpha_sa1_1}_{alpha_sb1_1}_m{alpha_sm1_1}_{alpha_sm2_1}_T{T}_step{step}_t{end}_norm_200.npy")

    plt.rcParams["figure.figsize"] = [16,9]

    fig, axes = plt.subplots(nrows=10, ncols=10, sharex=False)

    for i in range(10):
        
        for j in range(10):
            # axes[i,j].plot(t_data, e_all_c[10*i+j])
            axes[i,j].plot(t_data, e_all_p[100+10*i+j])
            
            # axes[i,j].plot(t_data, e_all_c[3*i+j], color="tab:green", label = "Conventional")
            # axes[i,j].plot(t_data, e_all_p[3*i+j], color="tab:red", label = "Proposed")
            # axes[i,j].legend()
            # axes[i,j].axis("off")
            axes[i,j].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            # axes[i,j].grid()
            axes[i,j].set_title(f"{100+10*i+j}", loc="right", pad=-10.0)

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.01, wspace=0.01)
    plt.suptitle(dir_base_0 + f"m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_norm_s100-199")
    
    os.makedirs(dir_base_0 + "fig/", exist_ok=True)
    plt.savefig(dir_base_0 + f"fig/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_norm_s100-199.png")

    # plt.show()

if __name__ == '__main__':
    
    start = time.perf_counter()
    
    print(datetime.datetime.now())
    
    use_cpu = 20
    
    print(f"use cpu core {use_cpu}/{cpu_count()}")

    init = []
    
    list_dirbase_0 = ["./data/b_no/","./data/s_no/","./data/s_bzd/","./data/s_bzdm/"]

    list_alpha_lambda = [0.0,0.3,0.5]
    
    list_alpha_sab1 = [1.0,2.0,3.0]
    
    for i in list_dirbase_0:
        
        for j in list_alpha_lambda:
            
            for k in list_alpha_sab1:
                
                init.append((i,j,k))
        

    with Pool(use_cpu) as p:
        
        r = p.starmap(func=main,iterable=init)
        
    print(time.perf_counter() - start)