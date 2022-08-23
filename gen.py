import pandas as pd
import numpy as np
import scipy.sparse as sp
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_network(Z0, Z1, Z2):
    N = PARAM['N']
    A_dt = np.eye(N,N)

    for i in range(len(Z0)):
        for j in range(i+1, len(Z0)):
            zi0, zi1, zi2 = np.array(Z0[i]), np.array(Z1[i]), np.array(Z2[i])
            zj0, zj1, zj2 = np.array(Z0[j]), np.array(Z1[j]), np.array(Z2[j])
            logit = np.exp(PARAM['alpha0'] - PARAM['alpha1'] * (
                np.sqrt(np.square(zi0-zj0)) + np.sqrt(np.square(zi1-zj1)) + np.sqrt(np.square(zi2-zj2))
            )) * PARAM['network_density']
            friend = np.random.binomial(1, logit / (1 + logit))
            A_dt[i][j], A_dt[j][i] = friend, friend
    network_density.append(((A_dt.sum() - N) * 2 / (N * (N - 1))))
    return pd.DataFrame(A_dt)

def generate_y(Z0, Z1, Z2):
    eps = np.random.normal(0, 0.5, size=PARAM['N'])
    T = np.random.binomial(1, 0.5, size=PARAM['N'])
    Z0, Z1, Z2 = np.array(Z0), np.array(Z1), np.array(Z2)
    y = Z0 + Z1 + Z2 + T*Z0 + 0.5*T*Z1 + eps
    return T, y

def main():
    cluster = np.random.randint(0,3,size=PARAM['N'])
    user_dt = pd.DataFrame(cluster, columns=['cluster'])
    user_dt = pd.get_dummies(user_dt['cluster'], prefix='Z')
    Z0, Z1, Z2 = user_dt['Z_0'], user_dt['Z_1'], user_dt['Z_2'] # covariates
    A = generate_network(Z0, Z1, Z2)
    A.to_csv('network.csv', index=False)  ## TODO: save files

    T, y = generate_y(Z0, Z1, Z2)
    T = pd.DataFrame(T, columns=['T'])
    y = pd.DataFrame(y, columns=['y'])
    dt_train = pd.concat([user_dt,T,y], axis=1)
    dt_train.to_csv('user.csv', index=False)

PARAM = {
    # network parameters
    'N': 100,   # num_nodes
    'network_density': 0.33,
    'alpha0': 0,
    'alpha1': 1,
}

seed = 11
if __name__ == "__main__":
    # dir_dt = ''   ## TODO: data path
    # if not os.path.isdir(dir_dt):
    #     os.makedirs(dir_dt)
    #
    # # repead experiment (if needed)
    network_density = []
    set_seed(seed)
    # for seed in range(11,111):
    #     set_seed(seed)
    #     print("generate data:",seed)
    #     main()
    main()
    print("average edge prob:", np.mean(network_density))