import csv
import time
import numpy as np
from tqdm import tqdm
import os

t = 0
end = 100
step = 0.0001
i = 0
t_data = []

start = time.time()

for i in tqdm(range(int(end/step))):
# while t < end:

    if i%10 == 0:

        t_data.append(t)

        # print(round(100*t/end, 1),"%",i,round(time.time()-start, 1),"s")

    t += step
    i += 1

# dir_base = "./data/bzd/"
# dir_base = "./data/bz/"
# dir_base = "./data/no/"
dir_base = "./data/"
# dir_base = "./"
os.makedirs(dir_base, exist_ok=True)
print(len(t_data))
np.savetxt(f"step{step}_t{end}.csv",t_data,delimiter = ",")