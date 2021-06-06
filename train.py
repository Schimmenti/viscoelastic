from os import pardir
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import argparse
from u_net_model import *
import torch.cuda
from torch.utils.data import DataLoader
from viscoset import *
from torch.utils.data.sampler import SubsetRandomSampler

dvc_cnt = torch.cuda.device_count()
device_names = [torch.cuda.get_device_name(idx)  for idx in range(dvc_cnt)]

devices = [torch.device("cuda:" + str(idx))  for idx in range(dvc_cnt)]


dvc = devices[0]

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='')
parser.add_argument('--output_dir', default='')
parser.add_argument('--dataset_list', default='')
parser.add_argument('--model_filename', default='')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=15)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--evaluate', type=int, default=0)
parser.add_argument('--regression', type=int, default=0)
parser.add_argument('--train_split',type=float, default=0.7)
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
dataset_list = args.dataset_list
model_filename = args.model_filename
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
regression = args.regression > 0
evaluate = args.evaluate > 0
train_split = args.train_split

print('Arguments parsed.')

np.random.seed(1204565)
torch.manual_seed(1204565)

print('Dataset loading...')

dataset = ViscoelasticDataset(dataset_list, input_dir, output_dir, input_mode=1, regression=regression)

# Creating data indices for training and test splits:
shuffle_dataset = True
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(train_split * dataset_size))
if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)

print('Dataset loaders created...')

print('Network creation...')

net = UNet(3,1)

try:
    net.load_state_dict(torch.load(model_filename))
    print('Model loading completed...')      
except:
    print('Training from scratch...')


if(evaluate):
    #we do not use GPU
    with torch.no_grad():
        net.eval()
        n_evals = 1
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            print('Evaluating batch n° %i...' % batch_index)
            if(batch_index >= n_evals):
                break
            y_out = net(x_batch)
            np.save('test_result_%i.npy' % batch_index, np.concatenate((x_batch.numpy()[np.newaxis,...],y_batch.numpy()[np.newaxis,np.newaxis,...],y_out.numpy()[np.newaxis,np.newaxis,...]),axis=0))
        exit(0)
net.to(dvc)


optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
if(regression):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.MSELoss()


net.train()
loss_history = []


print('Training...')

for epoch in range(epochs):
    print('Epoch: ', epoch)
    avg_loss = 0
    batch_counts = 0
    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        y_out = net(x_batch.to(dvc))
        loss = criterion(y_out, y_batch.to(dvc))
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        avg_loss += loss.item()
        optimizer.step()
        batch_counts += 1
    avg_loss /= batch_counts
    loss_history.append(avg_loss)
    if(epoch % 5 == 0 and model_filename != ""):
        print(avg_loss) 
        torch.save(net.state_dict(), model_filename)
        np.savetxt('train_loss.txt', np.array(loss_history))

#parser.add_argument('--batches_folder', default='vdep_batches/')
#parser.add_argument('--min_batch_idx', type=int, default=0)
#parser.add_argument('--max_batch_idx', type=int, default=9)
#parser.add_argument('--model_state_file')
#parser.add_argument('--lr', type=float, default=0.001)
#parser.add_argument('--batch_sz', type=int, default=15)
#parser.add_argument('--train_ratio', type=float, default=0.7)
#parser.add_argument('--epochs', type=int, default=500)
#parser.add_argument('--evaluate', type=int, default=0)
#parser.add_argument('--regression', type=int, default=0)

#print('Arguments loading.')
#args = parser.parse_args()
#folder = args.batches_folder
#min_idx = args.min_batch_idx
#max_idx = args.max_batch_idx
#model_state_file = args.model_state_file
#lr = args.lr
#batch_sz = args.batch_sz
#epochs = args.epochs
#evaluate = args.evaluate > 0
#train_ratio = args.train_ratio
#regression = args.regression > 0
#print('Folder with dataset: ', folder)
#print('Model file: ', model_state_file)



#x_train = []
#x_test = []
#y_train = []
#y_test = []
#
#print('Initial phase: dynamical dataset creation')
#
#for idx in range(min_idx,max_idx+1):
#    print('Loading batch %i' % idx)
#    temp = np.load(folder + ('batch_%i.npy' % idx), allow_pickle=True)
#    indices = np.random.choice(temp.shape[0],size=temp.shape[0], replace=False)
#    train_sz = int(train_rhttps://github.com/Schimmenti/viscoelasticatio*len(indices))
#    train_indices = indices[:train_sz]
#    test_indices = indices[train_sz:]
#    x_train.append(torch.from_numpy(temp[train_indices,3:6,...]))
#    x_test.append(torch.from_numpy(temp[test_indices,3:6,...]))
#    if(regression):
#        y_train.append(torch.from_numpy(temp[train_indices,7,...]))
#        y_test.append(torch.from_numpy(temp[test_indices,7,...]))
#    else:
#        y_train.append(torch.from_numpy(temp[train_indices,6,...]))
#        y_test.append(torch.from_numpy(temp[test_indices,6,...]))
#
#x_train = torch.cat(x_train, dim=0)
#x_test = torch.cat(x_test, dim=0)
#y_train = torch.cat(y_train, dim=0)
#y_test = torch.cat(y_test, dim=0)




#print("Data loading complete.")
#
#net = UNet(3, 1)
#
##torch.save(model.state_dict(), PATH)
#
#try:
#    net.load_state_dict(torch.load(model_state_file))
#    print('Model loading completed.')      
#except:
#    print('Could not load saved model state.')
#
#
#if(evaluate):
#       net.eval()
#       random_indices = torch.from_numpy(np.random.choice(x_test.size(0), size=10)) 
#       x_batch = x_test[random_indices]
#       y_batch = y_test[random_indices]
#       y_out = net(x_batch)
#       print(x_batch.shape,y_batch.shape,y_out.shape)
#       print('Evaluated')
#       np.save('test_x_res.npy', x_batch.detach().numpy())
#       np.save('test_y_res.npy', y_batch.detach().numpy())
#       np.save('test_y_out.npy', y_out.detach().numpy())
#       exit()


#net.to(device)
#
#
##training part
#
#
#optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#if(regression):
#    criterion = nn.BCEWithLogitsLoss()
#else:
#    criterion = nn.MSELoss()
#net.train()
#loss_history = []
#for epoch in range(epochs):
#   if(epoch % 10 == 0):
#       print(epoch)
#   perm = torch.randperm(x_train.size(0))
#   idx = perm[:batch_sz]
#   x_batch = x_train[idx].to(device)
#   y_batch = y_train[idx].to(device)
#   y_out = net(x_batch)   
#   loss = criterion(y_out, y_batch)
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#   
#   loss_history.append(loss.item())
#   del x_batch
#   del y_batch
#   if(epoch % 50 == 0 and model_state_file != ""):
#       print(loss.item())
#       torch.save(net.state_dict(), model_state_file)
#       np.savetxt('train_loss.txt', np.array(loss_history))
#
#
#
#
#
#