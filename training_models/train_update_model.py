from configuration.config import Configuration
from utils import Utils
import numpy as np
from data_extractor import DataExtractor
import torch
from  torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torchvision
import os
import copy
from training_models.base_train import BaseTrain
import glob
import pandas as pd
from datasets import DeltasDataset
import pathlib


class TrainUpdateModels(BaseTrain):
    def __init__(self, update_type,parametrs):
        super(TrainUpdateModels,self).__init__(parameters=parametrs)
        self.model_parameters=self.config['models']['update_models']
        self.csv_batch_size=self.model_parameters['csv_batch_size']
        self.num_update_models=self.num_train_update_samples if update_type=='train' else self.num_test_update_samples
        self.update_type=update_type
        self.folder_name=Utils.get_folder_name(self.update_type, self.shadow_in_channel, self.shadow_out_channels, self.shadow_fc_size, self.shadow_learning_rate, self.update_learning_rate,self.shadow_num_epochs, self.update_shadow_epochs)
        self.deltas_csv_filename=f'{self.num_update_models}_{self.update_type}_delts.csv'
        self.deltas_csv_directory = self.cache_name + 'deltas_csvs/' + self.folder_name
        self.deltas_csv_path=  self.deltas_csv_directory + self.deltas_csv_filename




    def get_update_dataset(self,probe_data, update_sets,shadow_model):
        if os.path.exists(self.deltas_csv_path):
            delta_dataset=DeltasDataset(csv_file=self.deltas_csv_path)
            print (f'complete csv for {self.update_type} deltas exists, returning {self.update_type} {self.num_update_models} deltas')
            return delta_dataset
        else:
            if os.path.isdir(self.deltas_csv_directory)==False: pathlib.Path(self.deltas_csv_directory).mkdir(parents=True)
            if len(os.listdir(self.deltas_csv_directory))>5:
                print(f'complete csv for {self.update_type} deltas doesnt exists, mergeing csvs and returning {self.update_type} {self.num_update_models} deltas')
                self.merge_csv()
                delta_dataset = DeltasDataset(csv_file=f'{self.deltas_csv_path}/{self.deltas_csv_filename}')
                return delta_dataset
            else:
                print(f'no deltas exists, training {self.update_type} {self.num_update_models} deltas and returning dataset')

                self.train_deltas(probe_data, update_sets,shadow_model)
                self.merge_csv()
                delta_dataset = DeltasDataset(csv_file=self.deltas_csv_path)
                print (f'finished training {self.update_type} {self.num_update_models} deltas and returning dataset')
                return delta_dataset

    def train_deltas(self,probe_data,update_sets,shadow_model):

        probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=len(probe_data),
                                                       shuffle=False)
        probe_x=next(iter(probe_loader))[0].to(self.device)
        probe_orignel_y=shadow_model(probe_x).view(-1)

        csv_batches=range(0, (update_sets.__len__()),self.csv_batch_size)

        for batch_start in csv_batches:
            delta_list=torch.zeros((self.csv_batch_size, probe_orignel_y.shape[0])).to(self.device)


            for i, update_set in enumerate(update_sets[batch_start:batch_start+self.csv_batch_size]):
                copied_model = copy.deepcopy(shadow_model).to(self.device)
                new_model=self.update_model(copied_model,update_set)

                delta_list[i]=new_model(probe_x).view(-1)

                if i%100==0: print (f'batch {batch_start} sample {i}')

            deltas_list = probe_orignel_y.view(1, -1) - delta_list
            deltas_list=deltas_list.cpu().detach().numpy()
            deltas_list=pd.DataFrame(deltas_list)
            #numpy.savetxt(f'{self.deltas_csv_path}/{self.update_type}_deltas_{batch_start}_{batch_start+self.csv_batch_size}.csv',deltas_list)
            deltas_list.to_csv(f'{self.deltas_csv_directory}/{self.update_type}_deltas_{batch_start}_{batch_start+self.csv_batch_size}.csv')


        return deltas_list

    def update_model(self, net, update_set):

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), self.update_learning_rate)
        update_loader = torch.utils.data.DataLoader(update_set, batch_size=64, num_workers=0)

        net.train()
        for epoch in range(self.update_shadow_epochs):
            for i, (images, labels) in enumerate(update_loader):
                optimizer.zero_grad()
                outputs = net(images.to(self.device))
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()

        return net

    def merge_csv(self):
        root = os.getcwd()
        os.chdir(self.deltas_csv_directory)
        all_filenames = [i for i in glob.glob('*.{}'.format('csv'))]
        combined_csv = pd.concat([pd.read_csv(f, index_col=0) for f in all_filenames])
        combined_csv.to_csv(self.deltas_csv_filename, index=False)
        os.chdir(root)



