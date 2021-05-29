import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from numba import jit
import scipy
import scipy.stats as stats
import argparse

Lx=512
Ly=512

parser = argparse.ArgumentParser()
parser.add_argument('--main_folder')
parser.add_argument('--folder')
parser.add_argument('--file_list')
parser.add_argument('--out_folder')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--data_file')
args = parser.parse_args()
main_folder = args.main_folder
folder = args.folder
flist = args.file_list
out_folder = args.out_folder
batch_size = args.batch_size
data_file = args.data_file
file_idxs = np.loadtxt(flist).astype('int')
table = np.loadtxt(data_file, delimiter=' ', skiprows=1)
print('Number of files: ', len(file_idxs))


current_batch = []
for cnt,idx in enumerate(file_idxs):
   try:
      input_data = np.fromfile(main_folder + ('events0_%i.dat' % idx), dtype='float32').reshape(1,3,Lx,Ly)
      input_data2 = np.fromfile(main_folder + ('events1_%i.dat' % idx), dtype='float32').reshape(1,3,Lx,Ly) 
      data = np.fromfile(folder +('distr0_%i.dat' % idx), dtype='float32')
      data = data.reshape(1,2,Lx,Ly)
      current_batch.append(np.concatenate([input_data,input_data2, data], axis=1))  
      if((cnt+1) % batch_size==0):
         print(cnt)
         np.save( out_folder + ('batch_%i.npy'%(cnt//batch_size)),  np.concatenate(current_batch, axis=0))
         current_batch = [] 
   except:
      np.save( out_folder + ('batch_%i.npy'%(1+cnt//batch_size)),  np.concatenate(current_batch, axis=0))
      break
if(len(current_batch)>0):
   np.save( out_folder + ('batch_%i.npy'%(1+cnt//batch_size)),  np.concatenate(current_batch, axis=0))



