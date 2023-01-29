# -*- coding: utf-8 -*-
# rfwnn
# 0 system
# D
# no wn
# no sign
# beta zeta D stop
# old s
# alpha_lambda D

from func_d import *
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import datetime
from numba import njit

# @njit
def main(n_seed: int):
    
    np.random.seed(n_seed)
    
    t = 0.0
    i = 0
    
    end = 100
    step = 0.0001

    alpha_w = 50.0 * np.identity(5)
    alpha_v = 20.0 * np.identity(75)
    alpha_a = 20.0 * np.identity(75)
    alpha_b = 20.0 * np.identity(75)
    alpha_beta = 0.001 * np.identity(5)
    alpha_zeta = 0.1
    alpha_lambda = 0.0
    alpha_d = 10.0 * np.identity(3)
    alpha_dk = 0.2 * np.identity(3)
    alpha_sa1 = 1.0
    alpha_sb1 = 1.0
    alpha_sm1 = 2.0
    alpha_sm2 = 0.6

    T = 1000

    p = np.array([4, 3, 1.5])
    l = np.array([0.4, 0.3, 0.2])
    g = 9.8
    
    zeta = 1.0
    omega = np.ones((5,1))
    beta = 0.1 * np.array([
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0]],
        dtype=np.float64
    )

    m = -0.01
    n = 1.01
    q = np.array([
        [m * 0.5, n * np.pi],
        [m * 0.5, n * np.pi],
        [m * 0.5, n * np.pi]]
    )
    # xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,0,0,0]]).reshape(-1,1)
    xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5]]).reshape(-1,1)
    xold = [xold0 for i_xold in range(T)]

    W = 50 * 2 * (np.random.rand(5,3) - 0.5)
    j_q = 1.0 * 0.5
    j_dq = 1.0 * np.pi
    # j_ddq = 2.0 * np.pi**2
    j_s = 0.1 * 1.0 * np.pi * np.sqrt(2)
    # j = np.array([[j_q,j_q,j_q,j_dq,j_dq,j_dq,j_q,j_q,j_q,j_dq,j_dq,j_dq,j_ddq,j_ddq,j_ddq]]).T
    j = np.array([[j_q,j_q,j_q,j_dq,j_dq,j_dq,j_q,j_q,j_q,j_dq,j_dq,j_dq,j_s,j_s,j_s]]).T
    v = j * 0.1 * 2 * (np.random.rand(15,5) - 0.5)
    a = (1/j) * 0.5 * 2 * (np.random.rand(15,5) - 0.5)
    b = j * 1 * 2 * (np.random.rand(15,5) - 0.5)
    D = np.ones((3,3))
    
    Wold = W.copy()
    vold = v.copy()
    aold = a.copy()
    bold = b.copy()
    Dold = D.copy()

    Woldold = W.copy()
    voldold = v.copy()
    aoldold = a.copy()
    boldold = b.copy()
    Doldold = D.copy()

    e_0 = []
    e_1 = []
    e_2 = []
    e_3 = []
    e_4 = []
    e_5 = []
    e_6 = []
    e_7 = []
    e_8 = []
    e_9 = []
    e_10 = []
    e_11 = []
    e_12 = []
    e_13 = []
    e_14 = []
    e_15 = []
    e_16 = []
    e_17 = []
    e_18 = []
    e_19 = []
    e_20 = []
    e_21 = []
    e_22 = []
    e_23 = []
    e_24 = []
    e_25 = []
    e_26 = []

    for i in tqdm(range(int(end/step))):
    # for i in range(int(end/step)):

        qd = qd_f(t)
        e = e_f(t,q)
        s = s_f(e)
        x = x_f(q,qd,s)
        xji = xji_f(x,xold[-T],v,a,b)
        A = A_f(xji,a,b)
        Aold = A_f(xold[-T], a,b)
        B = B_f(x,Aold,v,b)
        muji = muji_f(A)
        mu = mu_f(muji)
        omega = omega_f(v,a,b,W)
        y = y_f(muji,W)
        taus0 = taus0_f(s,beta,zeta,omega)
        taud = taud_f(s,taus0,y)
        tau = tau_f(taud, D)
        vk = vk_f(mu,muji,A,Aold,B,a)
        ak = ak_f(mu,muji,A,Aold,B,v,a,b,xold[-1])
        bk = bk_f(mu,muji,A,Aold,B,v,a,b)

        if zeta > 0.1:

            k_beta = np.linalg.norm(s) * alpha_beta @ omega
            k_zeta = -alpha_zeta * zeta
            k_D = alpha_d @ np.sign(taud) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

        else :

            k_beta = np.zeros((5,1))
            k_zeta = 0.0
            k_D = np.zeros((3,3))

        k1_q = system(t,q,tau,p,l,g)
        k2_q = system(t+step/2,q+(step/2)*k1_q,tau,p,l,g)
        k3_q = system(t+step/2,q+(step/2)*k2_q,tau,p,l,g)
        k4_q = system(t+step,q+step*k3_q,tau,p,l,g)

        xold.append(x.copy())

        Woldold = Wold.copy()
        voldold = vold.copy()
        aoldold = aold.copy()
        boldold = bold.copy()
        Doldold = Dold.copy()

        Wold = W.copy()
        vold = v.copy()
        aold = a.copy()
        bold = b.copy()
        Dold = D.copy()

        if i%10 == 0:

                e_0.append(e[0][0])
                e_1.append(e[1][0])
                e_2.append(e[2][0])
                e_3.append(e[0][1])
                e_4.append(e[1][1])
                e_5.append(e[2][1])
                e_6.append(s[0][0])
                e_7.append(s[1][0])
                e_8.append(s[2][0])
                e_9.append(tau[0][0])
                e_10.append(tau[1][0])
                e_11.append(tau[2][0])
                e_12.append(taud[0][0])
                e_13.append(taud[1][0])
                e_14.append(taud[2][0])
                e_15.append(y[0][0])
                e_16.append(y[1][0])
                e_17.append(y[2][0])
                e_18.append(taus0[0][0])
                e_19.append(taus0[1][0])
                e_20.append(taus0[2][0])
                e_21.append(D[0][0])
                e_22.append(D[1][1])
                e_23.append(D[2][2])
                e_24.append(beta.T @ omega)
                e_25.append(np.linalg.norm(s))
                e_26.append(np.linalg.norm(e[:,0:1]))

        q += (step / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        W += step * (alpha_w @ (mu.reshape(5,1) - vk.T @ v.T.reshape(-1,1) - ak.T @ a.T.reshape(-1,1) - bk.T @ b.T.reshape(-1,1)) @ s.T) + alpha_lambda * (Wold - Woldold)
        v += step * ((alpha_v @ vk @ W @ s).reshape(5,15).T) + alpha_lambda * (vold - voldold)
        a += step * ((alpha_a @ ak @ W @ s).reshape(5,15).T) + alpha_lambda * (aold - aoldold)
        b += step * ((alpha_b @ bk @ W @ s).reshape(5,15).T) + alpha_lambda * (bold - boldold)
        beta += step * k_beta
        zeta += step * k_zeta
        # D += step * k_D
        D += step * k_D + alpha_lambda * (Dold - Doldold)

        t += step
        i += 1

    e_all = [
        e_0,
        e_1,
        e_2,
        e_3,
        e_4,
        e_5,
        e_6,
        e_7,
        e_8,
        e_9,
        e_10,
        e_11,
        e_12,
        e_13,
        e_14,
        e_15,
        e_16,
        e_17,
        e_18,
        e_19,
        e_20,
        e_21,
        e_22,
        e_23,
        e_24,
        e_25,
        e_26,
    ]

    param_all = [v,a,b,W,D,beta,zeta]

    dir_base = "./data/bzdm/"
    os.makedirs(dir_base, exist_ok=True)
    np.save(dir_base + f"s{n_seed}_m{alpha_lambda}_a{alpha_sa1}_{alpha_sb1}_m{alpha_sm1}_{alpha_sm2}_T{T}_step{step}_t{end}_param_all.npy",param_all)
    np.savetxt(dir_base + f"s{n_seed}_m{alpha_lambda}_a{alpha_sa1}_{alpha_sb1}_m{alpha_sm1}_{alpha_sm2}_T{T}_step{step}_t{end}_e_all.csv",e_all,delimiter = ",")

if __name__ == '__main__':
    
    start = time.perf_counter()
    
    print(datetime.datetime.now())
    
    use_cpu = 5
    
    print(f"use cpu core {use_cpu}/{cpu_count()}")

    init = []

    for i in range(100):
        
        init.append((i,))

    with Pool(use_cpu) as p:
        
        r = p.starmap(func=main,iterable=init)
        
    print(time.perf_counter() - start)

# if __name__ == '__main__':
    
#     main(0)