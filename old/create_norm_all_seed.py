import csv
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
"""
e_0.append(e[0][0])

e_3.append(tau[0])
e_6.append(taud[0])
e_9.append(y[0])
e_12.append(taus0[0])
e_15.append(K@s[0])
e_18.append(taus1[0])
e_21.append(wn[0])

e_24.append(e[0][1])
e_27.append(s[0])
e_30.append(D[0][0])

e_33.append(beta.T @ omega)
"""

alpha_lambda = 0.0

alpha_wn0 = 100
alpha_wn1 = 10

alpha_s0 = 5.0
alpha_s1 = 5.0
alpha_s2 = 5.0

T = 1000
step = 0.0001
end = 100

end_plt = 100
start_plt = 0

dir_base = "./data/bzd0/"

# t_data = np.loadtxt(dir_base + f"step{step}_t{end}.csv")

# e_all_p = np.loadtxt(dir_base + f"p_bzd_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_0s0}_{alpha_0s1}_{alpha_0s2}_T{T}_step{step}_t{end}_e_all.csv",delimiter = ",")
# e_all_c = np.loadtxt(dir_base + f"c_bzd_s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_1s0}_{alpha_1s1}_{alpha_1s2}_T{T}_step{step}_t{end}_e_all.csv",delimiter = ",")

n = 100

# e0 = np.zeros((n,100000))
# e1 = np.zeros((n,100000))
# e2 = np.zeros((n,100000))

norm = np.zeros((n,100000))

# for i in range(n):
    
#     # e_all_p = np.loadtxt(dir_base + f"p_bzd_s{i}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_0s0}_{alpha_0s1}_{alpha_0s2}_T{T}_step{step}_t{end}_e_all.csv",delimiter = ",")
#     e_all_p = np.loadtxt(dir_base + f"/bz/p_s{i}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv")
    
#     e0[i] = e_all_p[0]
#     e1[i] = e_all_p[1]
#     e2[i] = e_all_p[2]

# e0_mean = np.mean(e0,axis=0)
# e0_abs = np.abs(e0)
# e0_abs_mean = np.mean(e0_abs,axis=0)

# e1_mean = np.mean(e1,axis=0)
# e1_abs = np.abs(e1)
# e1_abs_mean = np.mean(e1_abs,axis=0)

# e2_mean = np.mean(e2,axis=0)
# e2_abs = np.abs(e2)
# e2_abs_mean = np.mean(e2_abs,axis=0)

# for i in range(n):
    
#     e_all_p = np.loadtxt(dir_base + f"p_s{i}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv")
    
#     for j in range(100000):
        
#         e = np.array([
#             [e[i][j]],
#             [e0[i][j]],
#             [e0[i][j]]],
#         )
        
#         norm[i][j] = np.linalg.norm(e)

for i in tqdm(range(n)):
    
    # e_all_p = np.loadtxt(dir_base + f"s{i}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv",delimiter = ",")
    # e_all_p = np.loadtxt(dir_base + f"s{i}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv")
    
    # norm[i] = e_all_p[34]
    norm[i] = np.loadtxt(dir_base + f"s{i}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv",delimiter = ",")[34]

# norm_mean = np.mean(norm,axis=0)

# e_mean = [e0_mean,e1_mean,e2_mean,e0_abs_mean,e1_abs_mean,e2_abs_mean,norm_mean]

# print(e0_mean.shape)
# print(e0_abs_mean.shape)
# print(norm.shape)

# print(e_all_p[0]-e0_mean)

os.makedirs(dir_base, exist_ok=True)

# np.savetxt(dir_base + f"m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_e_all.csv",e_mean,delimiter = ",")
np.savetxt(dir_base + f"m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_norm.csv",norm,delimiter = ",")