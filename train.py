import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import argparse
from u_net_model import *
import torch.cuda

print('Model load complete.')

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

print('Arguments loading.')
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

print('Folder with dataset: ', folder)
print('Model file: ', model_state_file)

np.random.seed(1204565)
torch.manual_seed(1204565)

x_train = []
x_test = []
y_train = []
y_test = []

print('Initial phase: dynamical dataset creation')

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
loss_history = []
for epoch in range(epochs):
   if(epoch % 10 == 0):
       print(epoch)
   perm = torch.randperm(x_train.size(0))
   idx = perm[:batch_sz]
   x_batch = x_train[idx].to(device)
   y_batch = y_train[idx].to(device)
   y_out = net(x_batch)   
   loss = criterion(y_out, y_batch)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   
   loss_history.append(loss.item())
   del x_batch
   del y_batch
   if(epoch % 50 == 0):
       print(loss.item())
       torch.save(net.state_dict(), model_state_file)
       np.savetxt('train_loss.txt', np.array(loss_history))





