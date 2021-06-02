import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import argparse
from u_net_model import *
import torch.cuda
import matplotlib.pyplot as plt
import pickle

dvc_cnt = torch.cuda.device_count()

best_idx = 0
best_mem = 0

for idx in range(dvc_cnt):
   t = torch.cuda.get_device_properties(0).total_memory
   if(t > best_mem):
      best_idx = idx


device = torch.device("cuda:" +str(best_idx))

parser = argparse.ArgumentParser()
parser.add_argument('--batches_folder', default='vdep_batches/')
parser.add_argument('--min_batch_idx', type=int, default=0)
parser.add_argument('--max_batch_idx', type=int, default=9)
parser.add_argument('--model_state_file')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_sz', type=int, default=5)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--evaluate', type=int, default=0)

args = parser.parse_args()
folder = args.batches_folder
min_idx = args.min_batch_idx
max_idx = args.max_batch_idx
model_state_file = args.model_state_file
lr = args.lr
batch_sz = args.batch_sz
epochs = args.epochs
evaluate = args.evaluate > 0
train_ratio = args.train_ratio

#we need here to load the desired files





#try:
#    x_data = np.load(folder + 'x_data.npy',allow_pickle=True)    
#    y0_data = np.load(folder + 'y0_data.npy',allow_pickle=True)
#    y1_data = np.load(folder + 'y1_data.npy',allow_pickle=True)
#except:    
#    x_data = []
#    y0_data = []
#    y1_data = []
#    for idx in range(min_idx,max_idx+1):
#        temp = np.load(folder + ('batch_%i.npy' % idx), allow_pickle=True)
#        x_data.append(temp[:,3:6,...])
#        y0_data.append(temp[:,6,...])
#        y1_data.append(temp[:,7,...])
#    x_data = np.concatenate(x_data,axis=0)
#    y0_data = np.concatenate(y0_data, axis=0)
#    y1_data = np.concatenate(y1_data,axis=0)
#    np.save(folder + 'x_data.npy',x_data)
#    np.save(folder + 'y0_data.npy',y0_data)
#    np.save(folder + 'y1_data.npy',y1_data)

#x_data = torch.from_numpy(x_data)
#y0_data = torch.from_numpy(y0_data)
#y1_data = torch.from_numpy(y1_data)



np.random.seed(1204565)
torch.manual_seed(1204565)

x_train = []
x_test = []
y_train = []
y_test = []

for idx in range(min_idx,max_idx+1):
    print('Loading batch %i' % idx)
    temp = np.load(folder + ('batch_%i.npy' % idx), allow_pickle=True)
    indices = np.random.choice(temp.shape[0],size=temp.shape[0], replace=False)
    train_sz = int(train_ratio*len(indices))
    train_indices = indices[:train_sz]
    test_indices = indices[train_sz:]
    x_train.append(torch.from_numpy(temp[train_indices,3:6,...]))
    x_test.append(torch.from_numpy(temp[test_indices,3:6,...]))
    y_train.append(torch.from_numpy(temp[train_indices,6,...]))
    y_test.append(torch.from_numpy(temp[test_indices,6,...]))

x_train = torch.cat(x_train, dim=0)
x_test = torch.cat(x_test, dim=0)
y_train = torch.cat(y_train, dim=0)
y_test = torch.cat(y_test, dim=0)


#tot_sz = x_data.shape[0]
#train_sz = int(args.train_ratio*tot_sz)
#test_sz = tot_sz - train_sz
#
#try:
#   x_train = np.load(folder+'x_train.npy', allow_pickle=True)
#   x_test = np.load(folder+'x_test.npy', allow_pickle=True)
#   y_train = np.load(folder+'y_train.npy', allow_pickle=True)
#   y_test = np.load(folder+'y_test.npy', allow_pickle=True)
#except:
#   indices = np.random.choice(tot_sz,size=tot_sz, replace=False)
#   train_indices = indices[:train_sz]
#   test_indices = indices[train_sz:]
#   x_train = x_data[train_indices]
#   x_test = x_data[test_indices]
#   y_train = y0_data[train_indices]
#   y_test = y0_data[test_indices]
#   np.save(folder+'x_train.npy', x_train)
#   np.save(folder+'x_test.npy', x_test)
#   np.save(folder+'y_train.npy', y_train)
#   np.save(folder+'y_test.npy', y_test)
#
#
#x_train = torch.from_numpy(x_train)
#x_test = torch.from_numpy(x_test)
#y_train = torch.from_numpy(y_train)
#y_test = torch.from_numpy(y_test)
#
#
#model loading



print("Data loading complete.")

net = UNet(3, 1)

#torch.save(model.state_dict(), PATH)

try:
    net.load_state_dict(torch.load(model_state_file))
    print('Model loading completed.')      
except:
    print('Could not load saved model state.')


if(evaluate):
       net.eval()
       random_indices = torch.from_numpy(np.random.choice(x_test.size(0), size=10)) 
       x_batch = x_test[random_indices]
       y_batch = y_test[random_indices]
       y_out = net(x_batch)
       print(x_batch.shape,y_batch.shape,y_out.shape)
       print('Evaluated')
       np.save('test_x_res.npy', x_batch.detach().numpy())
       np.save('test_y_res.npy', y_batch.detach().numpy())
       np.save('test_y_out.npy', y_out.detach().numpy())
       exit()


net.to(device)


#training part


optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()
net.train()
for epoch in range(epochs):
   print(epoch)
   perm = torch.randperm(x_train.size(0))
   idx = perm[:batch_sz]
   x_batch = x_train[idx].to(device)
   y_batch = y_train[idx].to(device)
   y_out = net(x_batch)   
   print(y_batch.shape, y_out.shape)
   loss = criterion(y_out, y_batch)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   print(loss.item())
   del x_batch
   del y_batch
   if(epoch % 10 == 0):
      torch.save(net.state_dict(), model_state_file)      






