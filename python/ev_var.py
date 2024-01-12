import numpy as np
import pickle

with np.load('ev_var.npz') as data:
    kEvMatrix = data['ev']
    kDevMatrix = data['dev']
