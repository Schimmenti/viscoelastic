import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import argparse
import u_net_model as unet
import viscoset as vis
import torch.cuda

from torch.utils.data.sampler import SubsetRandomSampler

print('Initialization',flush=True)

dvc = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='')
parser.add_argument('--output_dir', default='')
parser.add_argument('--dataset_list', default='')
parser.add_argument('--model_filename', default='')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--regression', type=int, default=0)
parser.add_argument('--train_split',type=float, default=0.7)


#parser.add_argument('--batches_epoch', type=int, default=20)
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
dataset_list = args.dataset_list
model_filename = args.model_filename
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
regression = args.regression > 0
train_split = args.train_split


trailing_name = 'bs=%i_lr=%f_regr=%i' % (batch_size, lr, args.regression)

if(model_filename == ''):
    model_filename = 'vdep_unet_' + trailing_name + '.dict'


print('Arguments parsed.',flush=True)
print('The model file name is: ', model_filename)

np.random.seed(1204565)
torch.manual_seed(1204565)

print('Dataset loading...',flush=True)

dataset = vis.ViscoelasticDataset(dataset_list, input_dir, output_dir, input_mode=1, regression=regression)

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

print('Dataset loaders created...',flush=True)

print('Network creation...',flush=True)

net = unet.UNet(3,1)

try:
    net.load_state_dict(torch.load(model_filename))
    print('Model loading completed...',flush=True)      
except:
    print('Training from scratch...',flush=True)


net.to(dvc)



#learning part


optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
if(regression):
    criterion = nn.MSELoss()
else:
    criterion = nn.BCEWithLogitsLoss()


net.train()
loss_history = []


print('Training...',flush=True)

for epoch in range(epochs):
    
    avg_loss = 0
    batch_counts = 0
    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        y_out = net(x_batch.to(dvc))
        loss = criterion(y_out, y_batch.to(dvc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        batch_counts += 1
    avg_loss /= batch_counts
    print('Epoch: ', epoch,flush=True)
    print('Loss: ', avg_loss, flush=True)
    loss_history.append(avg_loss)
    if(epoch % 5 == 0):
        torch.save(net.state_dict(), model_filename)
        np.savetxt('train_loss_' + trailing_name + '.txt', np.array(loss_history))