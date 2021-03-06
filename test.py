import torch
import torch.nn as nn
from torch.nn.modules.loss import TripletMarginLoss
import torch.optim as optim
import torch.functional as F
import numpy as np
import argparse
import u_net_model as unet
import viscoset as vis
import torch.cuda
import pickle as pkl
from torch.utils.data.sampler import SubsetRandomSampler

print('Initialization',flush=True)

if(torch.cuda.is_available()):
    print('Using CUDA', flush=True)
    dvc = torch.device('cuda')
else:
    dvc = torch.device('cpu')
    print('Using CPU', flush=True)


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='')
parser.add_argument('--output_dir', default='')
parser.add_argument('--dataset_list', default='')
parser.add_argument('--model_filename', default='')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--n_batches', type=int, default=3)
parser.add_argument('--regression', type=int, default=0)
parser.add_argument('--train_split',type=float, default=0.7)

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
dataset_list = args.dataset_list
model_filename = args.model_filename
batch_size = args.batch_size
n_batches = args.n_batches
train_split = args.train_split
regression = args.regression > 0
print('Arguments parsed.',flush=True)

np.random.seed(1204565)
torch.manual_seed(1204565)

print('Dataset loading...',flush=True)

dataset = vis.ViscoelasticDataset(dataset_list, input_dir, output_dir, input_mode=1, regression=regression)

# Creating data indices for training and test splits:
shuffle_dataset = True
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(train_split * dataset_size))
split2 = int(0.6*split)
train_indices, test_indices = indices[:split], indices[split:]
train_indices, validation_indices = train_indices[:split2], train_indices[split2:]

# Creating PT data samplers and loaders:
test_sampler = SubsetRandomSampler(test_indices)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=test_sampler)


print('Dataset loaders created...',flush=True)

print('Network creation...',flush=True)

net = unet.UNet(3,1)

if(torch.cuda.is_available()):
    checkpoint = torch.load(model_filename)
    
else:
    checkpoint = torch.load(model_filename, map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['state_dict'])
#net = nn.DataParallel(net)
net.to(dvc)

net.eval()

print('Evaluating...',flush=True)


with torch.no_grad():
    for batch_index, (x_batch, y_batch) in enumerate(test_loader):
        if(batch_index == n_batches):
            break
        y_out = net(x_batch.to(dvc))
        with open('test_results_%i.pkl' %batch_index, 'wb') as f:
            if(torch.cuda.is_available()):
                pkl.dump((x_batch.cpu().numpy(), y_batch.cpu().numpy(), y_out.cpu().numpy()),f)
            else:
                pkl.dump((x_batch.numpy(), y_batch.numpy(), y_out.numpy()),f)
        