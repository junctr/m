import csv
import numpy as np
from matplotlib import pyplot as plt
"""

"""
alpha_lambda0 = 0.0
alpha_lambda1 = 0.0

alpha_wn0 = 100
alpha_wn1 = 10

alpha_0s0 = 5.0
alpha_0s1 = 5.0
alpha_0s2 = 5.0

alpha_1s0 = 5.0
alpha_1s1 = 5.0
alpha_1s2 = 5.0

T = 1000
step = 0.0001
end = 100

end_plt = 100
start_plt = 0


dir_base0 = "./data/bzd/"
dir_base1 = "./data/no/"

t_data = np.loadtxt(f"./data/step{step}_t{end}.csv",delimiter = ",")

e_all_p = np.loadtxt(dir_base0 + f"m{alpha_lambda0}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_0s0}_{alpha_0s1}_{alpha_0s2}_T{T}_step{step}_t{end}_norm.csv",delimiter = ",")
e_all_c = np.loadtxt(dir_base1 + f"m{alpha_lambda1}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_1s0}_{alpha_1s1}_{alpha_1s2}_T{T}_step{step}_t{end}_norm.csv",delimiter = ",")

fig, axes = plt.subplots(nrows=10, ncols=10, sharex=False)

for i in range(10):
    
    for j in range(10):
        axes[i,j].plot(t_data, e_all_c[3*i+j])
        axes[i,j].plot(t_data, e_all_p[10*i+j])
        
        # axes[i,j].plot(t_data, e_all_c[3*i+j], color="tab:green", label = "Conventional")
        # axes[i,j].plot(t_data, e_all_p[3*i+j], color="tab:red", label = "Proposed")
        # axes[i,j].legend()
        
        axes[i,j].grid()

# plt.savefig(f"abrfwnn/data_test/s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_all.png")

# plt.plot(t_data, e_all_p)

plt.show()
