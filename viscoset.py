import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ViscoelasticDataset(Dataset):
    def __init__(self,dataset_list, input_dir, output_dir, input_mode=1,   regression=False, nfeatures=3, nrows=512, ncols=512):
        self.events_list = np.loadtxt(dataset_list, dtype=int)
        self.dset_length = len(self.events_list)
        self.input_dir = input_dir
        self.output_dir = output_dir
        if(input_mode == 0):
            self.input_format = "events0_%i.dat"
        else:
            self.input_format = "events1_%i.dat"
        self.output_format = "distr0_%i.dat"
        self.use_regression = regression
        self.nfeatures=nfeatures
        self.nrows = nrows
        self.ncols = ncols

    def __len__(self):
        return self.dset_length

    def __getitem__(self, idx):
        event_id = self.events_list[idx]
        input_filename = os.path.join(self.input_dir, self.input_format % event_id)
        output_filename = os.path.join(self.output_dir, self.output_format % event_id)
        input_data = torch.from_numpy(np.fromfile(input_filename, dtype='float32').reshape(self.nfeatures, self.nrows, self.ncols))
        output_data = np.fromfile(output_filename, dtype='float32').reshape(2, self.nrows, self.ncols)
        if(self.use_regression):
            output_data = torch.from_numpy(output_data[1,...]/ np.max(output_data[1,...]))
        else:
            output_data = torch.from_numpy(output_data[0,...])
        return input_data, output_data
