import csv
import numpy as np
from matplotlib import pyplot as plt
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
n_seed = 0

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

t_data = np.loadtxt(f"./data/step{step}_t{end}.csv",delimiter = ",")

e_all_p = np.load(dir_base_0 + f"s{n_seed}_m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_e_all.npy")
# e_all_c = np.load(dir_base_1 + f"s{n_seed}_m{alpha_lambda_1}_a{alpha_sa1_1}_{alpha_sb1_1}_m{alpha_sm1_1}_{alpha_sm2_1}_T{T}_step{step}_t{end}_e_all.npy")

plt.rcParams["figure.figsize"] = [16,9]

fig, axes = plt.subplots(nrows=9, ncols=3, sharex=False)

for i in range(9):
    
    for j in range(3):
        # axes[i,j].plot(t_data, e_all_c[3*i+j])
        axes[i,j].plot(t_data, e_all_p[3*i+j])
        
        # axes[i,j].plot(t_data, e_all_c[3*i+j], color="tab:green", label = "Conventional")
        # axes[i,j].plot(t_data, e_all_p[3*i+j], color="tab:red", label = "Proposed")
        # axes[i,j].legend()
        
        axes[i,j].grid()

# plt.savefig(f"abrfwnn/data_test/s{n_seed}_m{alpha_lambda}_wn{alpha_wn0}_{alpha_wn1}_s{alpha_s0}_{alpha_s1}_{alpha_s2}_T{T}_step{step}_t{end}_all.png")

plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, hspace=0.01, wspace=0.01)
plt.show()
