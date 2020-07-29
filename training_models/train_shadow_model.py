from configuration.config import Configuration
from utils import Utils
import numpy as np
from data_extractor import DataExtractor
import torch
from  torch.utils.data.sampler import SubsetRandomSampler
from models import ShadowModel, Generator, Encoder, Discriminator
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datetime import datetime
import os
import copy
from training_models.base_train import BaseTrain


class TrainShadow(BaseTrain):
    def __init__(self, parameters):
        super(TrainShadow,self).__init__(parameters=parameters)

        self.model_parameters=self.config['models']['shadow_model']
        self.batch_size=self.model_parameters['batch_size']
        self.shadow_logdir = self.log_dir + f'/shadow_model/{self.time}'
        self.model_name='shadow_model'
        self.model_file_name=Utils().get_model_name(self.model_name, self.shadow_in_channel, self.shadow_out_channels, self.shadow_fc_size, self.shadow_learning_rate, self.shadow_num_epochs)
        self.model_path = self.cache_name + self.model_file_name



    def get_trained_model(self, train_data, test_data):
        if os.path.exists(self.model_path):
            print ('shadow model already trained, loading')
            return torch.load(self.model_path, map_location=torch.device(self.device))
        else:
            print ('shadow model not trained, starting to train')
            return self.train_shadow(train_data, test_data)


    def train_shadow(self, train_data, test_data):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,shuffle=False)

        model=ShadowModel(self.shadow_in_channel, self.shadow_out_channels, self.shadow_fc_size).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.shadow_learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)

        loss_list = []
        accuracy_list = []

        print("Starting training...\n")
        writer = SummaryWriter(self.shadow_logdir, filename_suffix=self.time)
        # iterate over train set:

        training_loss = 0.0
        training_accuracy=0.0
        for epoch in range(self.shadow_num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):


                optimizer.zero_grad()
                outputs = model(images.to(self.device))
                loss = criterion(outputs, labels.to(self.device))
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                train_predicted = torch.max(outputs.data, 1)[1]
                train_total = labels.size(0)
                train_correct = (train_predicted == labels.to(self.device)).sum().item()
                training_accuracy+=(100 * train_correct / train_total)

                if i % 10 == 9:
                    writer.add_scalar('training loss',
                                      training_loss / 10,
                                      epoch * len(train_loader) + i)


                    writer.add_scalar('training accuracy',
                                      training_accuracy / 10,
                                      epoch * len(train_loader) + i)

                    training_loss=0.0
                    training_accuracy = 0.0


            scheduler.step()

            correct = 0
            total = 0

            # Iterate through valid dataset for accuracy
            model.eval()
            for images, labels in test_loader:
                outputs = model(images.to(self.device))
                valid_loss = criterion(outputs, labels.to(self.device))
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()
            accuracy = 100 * correct / float(total)
            # store loss and iteration
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)

            writer.add_scalar('test loss',loss.data,epoch)

            writer.add_scalar('test accuracy', accuracy, epoch)

            writer.close()

            print('Epoch: {}  Loss: {:.3f}  Accuracy: {:.3f} %'.format(epoch, loss.data.item(), accuracy))

        if os.path.isdir(self.cache_name)==False: os.mkdir(self.cache_name)
        torch.save(model,self.model_path)
        return model
