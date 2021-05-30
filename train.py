import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import argparse
from u_net_model import *
import torch.cuda

print('Number of CUDA devices: ')
print(torch.cuda.device_count())

dvc = torch.get_device('cuda') if torch.cuda.is_available() else torch.get_device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batches_folder', default='vdep_batches/')
parser.add_argument('--min_batch_idx', type=int, default=0)
parser.add_argument('--max_batch_idx', type=int, default=9)
parser.add_argument('--model_state_file')
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()
folder = args.batches_folder
min_idx = args.min_batch_idx
max_idx = args.max_batch_idx
model_state_file = args.model_state_file
lr = args.lr
try:
    x_data = np.load(folder + 'x_data.npy',allow_pickle=True)
    
    y0_data = np.load(folder + 'y0_data.npy',allow_pickle=True)
    y1_data = np.load(folder + 'y1_data.npy',allow_pickle=True)
except:    
    x_data = []
    y0_data = []
    y1_data = []
    for idx in range(min_idx,max_idx+1):
        temp = np.load(folder + ('batch_%i.npy' % idx), allow_pickle=True)
        x_data.append(temp[:,0:3,...])
        y0_data.append(temp[:,3,...])
        y1_data.append(temp[:,4,...])
    x_data = np.concatenate(x_data,axis=0)
    y0_data = np.concatenate(y0_data, axis=0)
    y1_data = np.concatenate(y1_data,axis=0)
    np.save(folder + 'x_data.npy',x_data)
    np.save(folder + 'y0_data.npy',y0_data)
    np.save(folder + 'y1_data.npy',y1_data)

x_data = torch.from_numpy(x_data)
y0_data = torch.from_numpy(y0_data)
y1_data = torch.from_numpy(y1_data)

net = UNet(3, 2)

#torch.save(model.state_dict(), PATH)

try:
    net.load_state_dict(torch.load(model_state_file))
    #net.eval()
except:
    print('Could not load saved model state.')

optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()









