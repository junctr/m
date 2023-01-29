import csv
import numpy as np
from matplotlib import pyplot as plt
"""
e_mean = [e0_mean,e1_mean,e2_mean,e0_abs_mean,e1_abs_mean,e2_abs_mean,norm_mean]
"""
alpha_lambda_0 = 0.0
alpha_sa1_0 = 1.0
alpha_sb1_0 = 1.0
alpha_sm1_0 = 2.0
alpha_sm2_0 = 0.6
dir_base_0 = "./data/s_bzdm/"

alpha_lambda_1 = 0.0
alpha_sa1_1 = 1.0
alpha_sb1_1 = 1.0
alpha_sm1_1 = 2.0
alpha_sm2_1 = 0.6
dir_base_1 = "./data/no/"

T = 1000
step = 0.0001
end = 100

end_plt = 100
start_plt = 0

t_data = np.loadtxt(f"./data/step{step}_t{end}.csv",delimiter = ",")

e_all_p = np.load(dir_base_0 + f"m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_mean.npy")
e_all_c = np.load(dir_base_1 + f"m{alpha_lambda_1}_a{alpha_sa1_1}_{alpha_sb1_1}_m{alpha_sm1_1}_{alpha_sm2_1}_T{T}_step{step}_t{end}_mean.npy")

plt.plot(t_data, e_all_c[6])
plt.plot(t_data, e_all_p[6])

# plt.savefig(f"abrfwnn/data_test/s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_all.png")

plt.show()
