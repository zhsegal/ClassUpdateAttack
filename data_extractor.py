from configuration.config import Configuration
from utils import Utils
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Subset
import numpy as np
import os

class DataExtractor():
    def __init__(self):
        self.config=Configuration().get_config()
        self.transformer_dict=self.config['preprocessing']['transformers']
        self.shadow_train_size=self.config['models']['sample_sizes']['shadow_train']
        self.probe_size=self.config['models']['sample_sizes']['probe_size']
        self.update_size=self.config['models']['sample_sizes']['update_size']
        self.num_train_update_samples = self.config['models']['sample_sizes']['num_train_update_samples']
        self.num_test_update_samples = self.config['models']['sample_sizes']['num_test_update_samples']

        self.data_file_path='data/raw_data_files/' + Utils().get_model_name(self.num_train_update_samples, self.num_test_update_samples, self.shadow_train_size)


    def get_data(self):
        if os.path.exists(self.data_file_path):
            print('data already calculated, loading')
            return torch.load(self.data_file_path)
        else:
            print('data not calculated, calculating')
            return self.calculate_data()

    def calculate_data(self):
        data_set = torchvision.datasets.CIFAR10
        transform=self.get_transformer()

        train_set = data_set(root='./data/cifar_raw', train=True,
                             download=True, transform=transform)
        test_set = data_set(root='./data/cifar_raw', train=False,
                            download=True, transform=transform)

        train_data, probe_data, train_update_sets, test_update_sets=self.split_data(train_set, test_set)

        torch.save((train_data, probe_data, train_update_sets, test_update_sets,  test_set), self.data_file_path)

        return train_data, probe_data, train_update_sets, test_update_sets,  test_set

    def split_data(self, train_set, test_set):
        shadow_train_indices=range(0,self.shadow_train_size)
        probe_indices=range(self.shadow_train_size, self.shadow_train_size + self.probe_size)


        train_data = Subset(train_set, shadow_train_indices)
        probe_data= Subset(train_set, probe_indices)

        train_update_sets = []
        for i in range(self.num_train_update_samples):
            indices = np.random.choice(range(self.shadow_train_size + self.probe_size, len(train_set)), self.update_size)
            train_update_sets.append(Subset(train_set, indices))

        test_update_sets = []
        for i in range(self.num_test_update_samples):
            indices = np.random.choice(range(0, len(test_set)),
                                       self.update_size)
            test_update_sets.append(Subset(test_set, indices))

        return train_data, probe_data,train_update_sets, test_update_sets


    def get_transformer(self):

        optional_transformers = {
            "crop_transform": transforms.RandomCrop(self.transformer_dict["crop_transform"][1],
                                                    self.transformer_dict["crop_transform"][2]),
            "flip_tansform": transforms.RandomHorizontalFlip(),

            "normalization_transform": transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        }

        optional_transformers_list = [optional_transformers[transformer] for transformer in optional_transformers.keys()
                                      if self.transformer_dict[transformer][0] == True]


        transform = transforms.Compose([transforms.ToTensor()] +  optional_transformers_list)
        return transform


