import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
import scipy
import scipy.stats as stats
import argparse

def load_params(filename):
    pdcit = {}
    f = open(filename)
    for l in f.readlines():
        a,b = l.split('=')
        pdcit[a] = float(b)
    return pdcit
 

parser = argparse.ArgumentParser()
parser.add_argument('--folder')
parser.add_argument('--minidx', type=int)
parser.add_argument('--maxidx', type=int)
args = parser.parse_args()
folder = args.folder
if(folder[-1] != '/'):
    folder += '/'
min_idx = args.minidx
max_idx = args.maxidx
pms = load_params(folder + 'params.txt')
pms['Lx'] = int(pms['Lx'])
pms['Ly'] = int(pms['Ly'])
data = np.loadtxt(folder + 'data.txt' , delimiter=' ', skiprows=1)


indices = []
for idx in range(min_idx, max_idx+1):
    temp  = np.loadtxt(folder + 'events_index_%i.txt' % idx)
    indices.append(temp.astype('int'))
indices = np.concatenate(indices)
start_idx = indices[np.argwhere(np.diff(np.insert(indices,0,0))>1).flatten()]
print(np.diff(start_idx))
exit()
dataset = []
for idx in range(min_idx, max_idx+1):
    events  = np.fromfile(folder + ('events_%i.dat' % idx), dtype=np.float32)
    n_saved_seqs = int(len(events)//(3*pms['Lx']*pms['Ly']))
    events = events.reshape(n_saved_seqs, 3, pms['Lx'],pms['Ly'])
    dataset.append(events)
dataset = np.concatenate(dataset, axis=0)

