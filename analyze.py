import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from numba import jit
import scipy
import scipy.stats as stats
import argparse
def bin_log(data, bins=50):
    if(type(bins) is int):
        binning = np.logspace(0,np.log10(np.max(data)), bins)
    else:
        binning = bins
    hist, edges = np.histogram(data,bins=binning, density=True)
    mask = hist > 0
    hist = hist[mask]
    edges = ((edges[0:-1]+edges[1:])*0.5)[mask]
    return hist, edges, binning

def load_params(filename):
    pdcit = {}
    f = open(filename)
    for l in f.readlines():
        a,b = l.split('=')
        pdcit[a] = float(b)
    return pdcit


parser = argparse.ArgumentParser()
parser.add_argument('--folder')
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--min_length', type=int, default=10)
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--max_n_events', type=int, default=10000)
#parser.add_argument('--minidx', type=int, default=0)
#parser.add_argument('--maxidx', type=int, default=0)
args = parser.parse_args()
folder = args.folder
#min_idx = args.minidx
#max_idx = args.maxidx

pms = load_params(folder + 'params.txt')
pms['Lx'] = int(pms['Lx'])
pms['Ly'] = int(pms['Ly'])
print(pms)



print('Loading...')
data = np.loadtxt(folder + 'data.txt' , delimiter=' ', skiprows=1)
seq_data = np.loadtxt(folder + 'a_seq.txt' , delimiter=' ', skiprows=1)

max_n_events = args.max_n_events


print('Filtering...')
#save all events with at least area min_area
min_area = args.min_area
valid_indices = np.argwhere(np.logical_and(data[:,2] >= min_area, data[:,3]==1)).flatten()
np.savetxt( 'events_to_save_area.txt', valid_indices[:min(max_n_events, len(valid_indices))], fmt='%i')



min_length = args.min_length
max_length = args.max_length
mainshock_idxs = np.argwhere(data[:,3]==0).flatten()
lengths = np.diff(mainshock_idxs)
mask_for_ms = np.argwhere(np.logical_and(lengths >= min_length, lengths <= max_length)).flatten()
good_ms = mainshock_idxs[mask_for_ms]
good_lens = lengths[mask_for_ms]

np.savetxt( 'events_to_save_mainshocks.txt', good_ms[:max_n_events],fmt='%i')
np.savetxt( 'events_to_save_seq.txt', np.concatenate([ np.arange(idx,idx+ln) for idx, ln in zip(good_ms, good_lens) ]).astype('int')[:max_n_events], fmt='%i')


exit()
