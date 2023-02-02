# -*- coding: utf-8 -*-
# rfwnn
# 0 system
# D
# no wn
# no sign
# no beta zeta D stop
# new s
# no alpha_lambda D

from func_d_s import *
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import datetime
from numba import njit

@njit
def sim(n_seed,alpha_lambda,alpha_sa1,alpha_sb1,alpha_sm1,alpha_sm2,T,step,end):
    
    np.random.seed(n_seed)
    
    t = 0.0
    i = 0
    i_10 = 0

    alpha_w = 50.0 * np.identity(5)
    alpha_v = 20.0 * np.identity(75)
    alpha_a = 20.0 * np.identity(75)
    alpha_b = 20.0 * np.identity(75)
    alpha_beta = 0.001 * np.identity(5)
    alpha_zeta = 0.1
    alpha_d = 10.0 * np.identity(3)
    alpha_dk = 0.2 * np.identity(3)

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
    # xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,0,0,0]]).copy().reshape(-1,1)
    xold0 = np.array([[m*0.5,m*0.5,m*0.5,n*np.pi,n*np.pi,n*np.pi,0,0,0,np.pi,np.pi,np.pi,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5,(1-n)*np.pi-m*2.5]]).copy().reshape(-1,1)
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
    D = np.zeros((3,3))
    
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

    e_all = np.zeros((27,int(end/step/10)))

    # for i in tqdm(range(int(end/step))):
    for i in range(int(end/step)):

        qd = qd_f(t)
        e = e_f(t,q)
        s = s_f(e,alpha_sa1,alpha_sb1,alpha_sm1,alpha_sm2)
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

        # if zeta > 0.015:

        #     k_beta = np.linalg.norm(s) * alpha_beta @ omega
        #     k_zeta = -alpha_zeta * zeta
        #     k_D = alpha_d @ np.sign(taud) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

        # else :

        #     k_beta = np.zeros((5,1))
        #     k_zeta = 0.0
        #     k_D = np.zeros((3,3))

        k_beta = np.linalg.norm(s) * alpha_beta @ omega
        k_zeta = -alpha_zeta * zeta
        k_D = alpha_d @ np.sign(taud) @ s.T - alpha_d @ alpha_dk @ D * np.linalg.norm(s)

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
                
                e_all[0][i_10] = e[0][0]
                e_all[1][i_10] = e[1][0]
                e_all[2][i_10] = e[2][0]
                e_all[3][i_10] = e[0][1]
                e_all[4][i_10] = e[1][1]
                e_all[5][i_10] = e[2][1]
                e_all[6][i_10] = s[0][0]
                e_all[7][i_10] = s[1][0]
                e_all[8][i_10] = s[2][0]
                e_all[9][i_10] = tau[0][0]
                e_all[10][i_10] = tau[1][0]
                e_all[11][i_10] = tau[2][0]
                e_all[12][i_10] = taud[0][0]
                e_all[13][i_10] = taud[1][0]
                e_all[14][i_10] = taud[2][0]
                e_all[15][i_10] = y[0][0]
                e_all[16][i_10] = y[1][0]
                e_all[17][i_10] = y[2][0]
                e_all[18][i_10] = taus0[0][0]
                e_all[19][i_10] = taus0[1][0]
                e_all[20][i_10] = taus0[2][0]
                e_all[21][i_10] = D[0][0]
                e_all[22][i_10] = D[1][1]
                e_all[23][i_10] = D[2][2]
                e_all[24][i_10] = (beta.T @ omega)[0][0]
                e_all[25][i_10] = np.linalg.norm(s)
                e_all[26][i_10] = np.linalg.norm(e[:,0:1])
                
                i_10 += 1

        q += (step / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        W += step * (alpha_w @ (mu.copy().reshape(5,1) - vk.T @ v.T.copy().reshape(-1,1) - ak.T @ a.T.copy().reshape(-1,1) - bk.T @ b.T.copy().reshape(-1,1)) @ s.T) + alpha_lambda * (Wold - Woldold)
        v += step * ((alpha_v @ vk @ W @ s).copy().reshape(5,15).T) + alpha_lambda * (vold - voldold)
        a += step * ((alpha_a @ ak @ W @ s).copy().reshape(5,15).T) + alpha_lambda * (aold - aoldold)
        b += step * ((alpha_b @ bk @ W @ s).copy().reshape(5,15).T) + alpha_lambda * (bold - boldold)
        beta += step * k_beta
        zeta += step * k_zeta
        D += step * k_D
        # D += step * k_D + alpha_lambda * (Dold - Doldold)

        t += step
        i += 1

    return e_all

# @njit
def main(n_seed,alpha_lambda,alpha_sab1):
    
    alpha_sa1 = alpha_sab1
    alpha_sb1 = alpha_sab1
    alpha_sm1 = 2.0
    alpha_sm2 = 0.6
    
    T = 1000
    step = 0.0001
    end = 100
    
    result = sim(n_seed,alpha_lambda,alpha_sa1,alpha_sb1,alpha_sm1,alpha_sm2,T,step,end)

    dir_base = "./data/s_no/"
    os.makedirs(dir_base, exist_ok=True)
    np.save(dir_base + f"s{n_seed}_m{alpha_lambda}_a{alpha_sa1}_{alpha_sb1}_m{alpha_sm1}_{alpha_sm2}_T{T}_step{step}_t{end}_e_all.npy",result)

if __name__ == '__main__':
    
    start = time.perf_counter()
    
    print(datetime.datetime.now())
    
    use_cpu = 20
    
    print(f"use cpu core {use_cpu}/{cpu_count()}")

    init = []

    list_alpha_lambda = [0.0,0.3,0.5]
    
    list_alpha_sab1 = [1.0,2.0,3.0]

    for i in range(120):
        
        for j in list_alpha_lambda:
            
            for k in list_alpha_sab1:
                
                init.append((i,j,k))

    with Pool(use_cpu) as p:
        
        r = p.starmap(func=main,iterable=init)
        
    print(time.perf_counter() - start)

# if __name__ == '__main__':
    
#     start = time.perf_counter()
    
#     print(datetime.datetime.now())
    
#     main(0)
    
#     print(time.perf_counter() - start)