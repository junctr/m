import numpy as np
import time
#import scipy.integrate as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
from multiprocessing import Pool, cpu_count
from numba import njit

def s0_f(e):

    s = -(5 * e)
    
    return s

def s1_f(e):

    s =  -(3 * e + 3 * np.abs(e)**(0.6) * np.sign(e))
    
    # s =  -(3 * np.abs(e)**(0.6) * np.sign(e))
    
    return s

e = np.linspace(-0.05,0.05,1000)

s0 = s0_f(e)

s1 = s1_f(e)

plt.plot(e, s0)
plt.plot(e, s1)

# plt.xlabel("time (s)")
# plt.ylabel(f"tracking error of link {n_e} (rad)")

# plt.xlim(start_plt,end_plt)
# plt.ylim(-40,40)
plt.legend()
plt.grid()

plt.show()