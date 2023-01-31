import csv
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
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

alpha_lambda_0 = 0.0
alpha_sa1_0 = 1.0
alpha_sb1_0 = 1.0
alpha_sm1_0 = 2.0
alpha_sm2_0 = 0.6
dir_base_0 = "./data/s_bzdm/"

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

e_mean = np.zeros((7,n,m))

for i in tqdm(range(n)):
    
    e_all_p = np.load(dir_base_0 + f"s{i}_m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_e_all.npy")
    
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

np.save(dir_base_0 + f"mean/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_mean.npy",e_mean)
np.save(dir_base_0 + f"mean/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_norm.npy",norm)