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


        

@jit(nopython=True,nogil=True)
def baksneppen(T,N,a):
    x = np.random.rand(N)
    j = np.arange(0,N)
    s = np.zeros(T)
    for t in range(T):
        i = np.argmin(x)
        start_j = i-a
        end_j = i + a + 1
        s[t] = x[i]
        for j in range(start_j,end_j):
            site = j % N
            x[site] = np.random.rand()
    return x,s

def load_params(filename):
    pdcit = {}
    f = open(filename)
    for l in f.readlines():
        a,b = l.split('=')
        pdcit[a] = float(b)
    return pdcit


parser = argparse.ArgumentParser()
parser.add_argument('--folder')
parser.add_argument('--minidx', type=int, default=0)
parser.add_argument('--maxidx', type=int, default=0)
args = parser.parse_args()
folder = args.folder
min_idx = args.minidx
max_idx = args.maxidx
pms = load_params(folder + 'params.txt')
pms['Lx'] = int(pms['Lx'])
pms['Ly'] = int(pms['Ly'])
print(pms)
print('Loading...')
#df = pd.read_csv(folder + 'data.txt', sep=' ')
#data = df.values
data = np.loadtxt(folder + 'data.txt' , delimiter=' ', skiprows=1)
seq_data = np.loadtxt(folder + 'a_seq.txt' , delimiter=' ', skiprows=1)
print('Filtering...')
#save all events with at least area min_area
min_area = 100
valid_indices = np.argwhere(np.logical_and(data[:,2] >= min_area, data[:,3]==1)).flatten()
np.savetxt( 'events_to_save_area.txt', valid_indices, fmt='%i')

min_length = 10
max_length = 100
mainshock_idxs = np.argwhere(data[:,3]==0).flatten()



exit()

for idx in range(min_idx, max_idx+1):
    events0 = np.fromfile(folder + ('events0_%i.dat' % idx), dtype=np.float32)
    events1 = np.fromfile(folder + ('events1_%i.dat' % idx), dtype=np.float32)
    events_ndx = np.loadtxt(folder + ('events_index_%i.txt' % idx)).astype('int')
    n_saved_seqs = int(len(events)//(3*pms['Lx']*pms['Ly']))
    events0 = events0.reshape(n_saved_seqs, 3, pms['Lx'],pms['Ly'])
    events1 = events1.reshape(n_saved_seqs, 3, pms['Lx'],pms['Ly'])

#for idx in range(min_idx,max_idx+1):
#    events = np.fromfile(folder + ('events_%i.dat' % idx), dtype=np.float32)
#    events_ndx = np.loadtxt(folder + ('events_index_%i.txt' % idx)).astype('int')
#    n_saved_seqs = int(len(events)//(3*pms['Lx']*pms['Ly']))
#    events = events.reshape(n_saved_seqs, 3, pms['Lx'],pms['Ly'])

#events to save creation











exit()

starting_idx = 500000

hist, edges, _ = bin_log(seq_data[:,0])
fig, ax = plt.subplots(figsize=(10,8))
plt.scatter(edges,hist)
tau=2.5
plt.plot(edges,edges**(-tau),lw=2,linestyle='dashed',color='black', label='$P(L_s) \\sim L_s^{-%1.2f}$'%tau )
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('$L_s$', fontsize=30)
plt.ylabel('$P(L_s)$', fontsize=30)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("len_sequence_plot.pdf")
plt.show()
hist, edges, _ = bin_log(seq_data[:,1])
fig, ax = plt.subplots(figsize=(10,8))
plt.scatter(edges,hist)
tau=2
plt.plot(edges,edges**(-tau),lw=2,linestyle='dashed',color='black', label='$P(A_s) \\sim A_s^{-%1.2f}$'%tau )
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('$A_s$', fontsize=30)
plt.ylabel('$P(A_s)$', fontsize=30)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("area_sequence_plot.pdf")
plt.show()
    
    
    
    
hist, edges, _ = bin_log(data[starting_idx:,0])
fig, ax = plt.subplots(figsize=(10,8))
plt.scatter(edges,hist)
tau=1.75
plt.plot(edges,edges**(-tau),lw=2,linestyle='dashed',color='black', label='$P(S) \\sim S^{-%1.2f}$'%tau )
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('$S$', fontsize=30)
plt.ylabel('$P(S)$', fontsize=30)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("plot.pdf")
plt.show()
    
        
hist, edges, _ = bin_log(data[starting_idx:,2])
fig, ax = plt.subplots(figsize=(10,8))
plt.scatter(edges,hist)
tau=2.0
plt.plot(edges,edges**(-tau),lw=2,linestyle='dashed',color='black', label='$P(A) \\sim A^{-%1.2f}$'%tau )
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('$A$', fontsize=30)
plt.ylabel('$P(A)$', fontsize=30)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig("area_plot.pdf")
plt.show()
