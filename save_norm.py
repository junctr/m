import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import datetime
from numba import njit
from matplotlib import pyplot as plt
"""
e_mean = [e0_mean,e1_mean,e2_mean,e0_abs_mean,e1_abs_mean,e2_abs_mean,norm_mean]
"""
alpha_lambda_0 = 0.0
alpha_sa1_0 = 3.0
alpha_sb1_0 = 3.0
alpha_sm1_0 = 2.0
alpha_sm2_0 = 0.6
dir_base_0 = "./data/s_no/"

alpha_lambda_1 = 0.0
alpha_sa1_1 = 3.0
alpha_sb1_1 = 3.0
alpha_sm1_1 = 2.0
alpha_sm2_1 = 0.6
dir_base_1 = "./data/b_no/"

T = 1000
step = 0.0001
end = 100

end_plt = 100
start_plt = 0

t_data = np.loadtxt(f"./data/step{step}_t{end}.csv",delimiter = ",")

e_all_p = np.load(dir_base_0 + f"mean/m{alpha_lambda_0}_a{alpha_sa1_0}_{alpha_sb1_0}_m{alpha_sm1_0}_{alpha_sm2_0}_T{T}_step{step}_t{end}_mean_remove.npy")
e_all_c = np.load(dir_base_1 + f"mean/m{alpha_lambda_1}_a{alpha_sa1_1}_{alpha_sb1_1}_m{alpha_sm1_1}_{alpha_sm2_1}_T{T}_step{step}_t{end}_mean_remove.npy")

plt.rcParams["figure.figsize"] = [16,11]
plt.rcParams["font.size"] = 30
plt.rcParams["lines.linewidth"] = 4

# plt.plot(t_data, e_all_c[6])
# plt.plot(t_data, e_all_p[6])

plt.figure()

plt.plot(t_data, e_all_c[6],color="green",label="conventional")
plt.plot(t_data, e_all_p[6],color="red",label="proposed")

plt.xlabel("Time (s)")
# plt.ylabel("norm($\it{e}$) (rad)")
plt.ylabel("$\it{||e||}$ (rad)")


# plt.xlim(0,100)
# plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.01, wspace=0.01)

plt.xlim(30,100)
plt.ylim(0,0.0002)
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.95, hspace=0.01, wspace=0.01)

plt.legend()
plt.grid()

# plt.show()

# os.makedirs("fig/", exist_ok=True)

# plt.savefig("fig/s.png")
plt.savefig("fig/s_30.png")
# plt.savefig("fig/"+dir_base_0+dir_base_1+"_30.png")
