from configuration.config import Configuration
from utils import Utils
import numpy as np
from data_extractor import DataExtractor
import torch

from  torch.utils.data.sampler import SubsetRandomSampler
from models import ShadowModel, Generator, Encoder, Discriminator,NewDiscriminator,NewGenerator
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datetime import datetime
import time
import os
from training_models.base_train import BaseTrain
from evaluate_model import EvaluateModel
import matplotlib.pyplot as plt
import pathlib
import random

class TrainGan(BaseTrain):
    def __init__(self,delta_set,update_sets,parameters):
        super(TrainGan, self).__init__(parameters=parameters)
        self.delta_sets=delta_set
        self.update_sets=update_sets

        self.GAN_parameters = self.config['models']['GAN']
        self.GAN_epoch_num=self.GAN_parameters['num_epochs']
        self.mu_dim=self.GAN_parameters['mu_dim']
        self.shadow_folder_name='gan_shadow_params_' + Utils.get_folder_name(self.shadow_in_channel, self.shadow_out_channels, self.shadow_fc_size, self.shadow_learning_rate, self.update_learning_rate, self.shadow_num_epochs, self.update_shadow_epochs)
        self.gan_folder_name= 'gan_params_' + Utils.get_folder_name(self.gan_loss_weight,self.generator_type,self.discriminator_type, self.noisy_label_change,self.discriminator_dropout,self.GAN_epoch_num,self.lr_gamma)

        self.gan_directory=self.cache_name + self.shadow_folder_name + self.gan_folder_name
        self.shadow_logdir = self.log_dir + f'/gan/{self.time}'

        self.generator_name='generator' + Utils().get_model_name(self.generator_leaky,self.generator_ngf)
        self.discriminator_name='discriminator' + Utils().get_model_name(self.dis_leaky,self.dis_ndf)
        self.encoder_name='encoder' + Utils().get_model_name(self.encoder_leky_reul,self.encoder_dropout)


        self.generator_path = self.gan_directory + self.generator_name
        self.discriminator_path = self.gan_directory + self.discriminator_name
        self.encoder_path = self.gan_directory + self.encoder_name
        self.pretrain_switch=self.GAN_parameters['pretrain_switch']


    def get_GAN(self):
        if os.path.exists(self.generator_path):
            print('GAN already trained, loading')
            return (torch.load(self.generator_path, map_location=torch.device(self.device)),
                    torch.load(self.encoder_path, map_location=torch.device(self.device)))
        else:
            print('GAN not trained, starting to train')
            return self.train_gan()

    def train_gan(self):
        #init models
        encoder,generator,discriminator=self.get_gan_models(self.pretrain_switch)



        #loss
        adversarial_loss = nn.BCELoss()
        adversarial_loss.cuda()
        bm_loss_mse = nn.MSELoss(reduction="sum")
        bm_loss_bce = nn.BCELoss(reduction="sum")

        # Optimizers
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002,betas=(0.5, 0.999))

        schedualer_E=torch.optim.lr_scheduler.StepLR(optimizer_E, step_size=10000, gamma=self.lr_gamma)
        schedualer_G=torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10000, gamma=self.lr_gamma)
        schedualer_D=torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10000, gamma=self.lr_gamma)


        mse_loss_g=0
        discriminator_loss_g=0
        total_loss_d = 0
        total_loss_g = 0
        gan_writer = SummaryWriter(self.shadow_logdir, filename_suffix=self.time)

        for epoch in range(self.GAN_epoch_num):
            for num, i in enumerate(random.sample(range(self.num_train_update_samples), self.num_train_update_samples)):
                if num%1000==0: print(f'finished sample {num}')
                #delta=self.delta_sets[i]
                delta=torch.Tensor(self.delta_sets[i]).to(self.device)

                update_set=self.update_sets[i]
                update_set_loader=torch.utils.data.DataLoader(update_set, batch_size=self.update_set_size)
                listed_update_set=next(iter(update_set_loader))[0].to(self.device)
                ## GENERATOR TRAIN
                optimizer_G.zero_grad()
                optimizer_E.zero_grad()

                real_labels = torch.FloatTensor(np.random.uniform(0.7, 1.2, (self.update_set_size, 1))).to(self.device)
                fake_labels = torch.FloatTensor(np.random.uniform(0, 0.3, (self.update_set_size, 1))).to(self.device)

                z=torch.randn(self.update_set_size, self.noise_dim, 1, 1, device=self.device)
                mu=encoder(delta)
                mu=self.add_dimension_to_mu(mu)
                generated_images=generator(mu.to(self.device), z)
                closest_images=self.find_closet_imgs(update_set, generated_images)
                output = discriminator(generated_images,mu)

                mse_loss=bm_loss_mse(closest_images,listed_update_set)
                mse_loss=mse_loss*self.gan_loss_weight

                discriminator_loss=bm_loss_bce(output, real_labels)
                generator_loss=mse_loss+discriminator_loss

                generator_loss.backward(retain_graph=True)
                optimizer_G.step()
                optimizer_E.step()

                schedualer_E.step()
                schedualer_G.step()
                #### DISCRIMIATOR TRAIN

                optimizer_D.zero_grad()

                #with noisy labels

                chance=np.random.binomial(1,self.noisy_label_change)

                if chance ==1:
                    real_labels_noisy=torch.cat([real_labels,torch.FloatTensor(np.random.uniform(0.7, 1.2, (3, 1))).to(self.device)])
                    fake_labels_noisy=torch.cat([fake_labels,torch.FloatTensor(np.random.uniform(0.0, 0.3, (3, 1))).to(self.device)])

                    listed_update_set_noisy=torch.cat([listed_update_set,generated_images[:3,:,:,:].detach() ])
                    generated_images_noisy=torch.cat([generated_images.detach(),listed_update_set[:3,:,:,:].detach() ])

                    outcome_real = discriminator(listed_update_set_noisy, torch.cat([mu.detach(),mu[:3,:].detach()]))
                    real_loss = adversarial_loss(outcome_real, real_labels_noisy)

                    outcome_fake = discriminator(generated_images_noisy, torch.cat([mu.detach(),mu[:3,:].detach()]))
                    fake_loss = adversarial_loss(outcome_fake, fake_labels_noisy)
                    d_loss = real_loss + fake_loss

                else:
                    outcome_real=discriminator(listed_update_set,mu.detach())
                    real_loss=adversarial_loss(outcome_real, real_labels)

                    outcome_fake=discriminator(generated_images.detach(),mu.detach())
                    fake_loss=adversarial_loss(outcome_fake, fake_labels)
                    d_loss=real_loss+fake_loss

                d_loss.backward()
                optimizer_D.step()
                schedualer_D.step()

                mse_loss_g+=mse_loss.item()
                discriminator_loss_g+=discriminator_loss.item()
                total_loss_d += d_loss.item()
                total_loss_g += generator_loss.item()



            total_loss_d /= num
            total_loss_g /= num
            mse_loss_g/= num
            discriminator_loss_g/= num

            EvaluateModel().evaluate(total_loss_d,mse_loss_g,discriminator_loss_g, epoch, gan_writer)
            lr='lr'
            print(f'{optimizer_D.param_groups[0][lr]}_learning_rate')

            print( "Epoch %d/%d D loss: %f G loss: %f time: %d"
                % (epoch, self.GAN_epoch_num, total_loss_d, total_loss_g, int(time.time())))

        if os.path.isdir(self.gan_directory)==False: pathlib.Path(self.gan_directory).mkdir(parents=True, exist_ok=True)
        torch.save(generator, self.generator_path)
        torch.save(discriminator, self.discriminator_path)
        torch.save(encoder, self.encoder_path)
        return generator, encoder




    def init_old_generator(self):

        #img_shape=32
        generator = Generator(self.mu_dim, self.noise_dim, self.generator_leaky,self.generator_ngf)
        generator.cuda()
        return generator

    def init_new_generator(self):

        generator = NewGenerator(self.mu_dim, self.noise_dim)
        generator.cuda()
        return generator

    def init_encoder(self):
        leaky_relu_parmas=self.encoder_leky_reul
        dropout_params=self.encoder_dropout

        encoder = Encoder(self.mu_dim,self.update_set_size,leaky_relu_parmas,dropout_params)
        encoder.cuda()

        return encoder

    def init_old_discriminator(self):
        discriminator=Discriminator(self.dis_leaky, self.dis_ndf)
        discriminator.cuda()
        return discriminator

    def init_new_discriminator(self):
        discriminator=NewDiscriminator(self.dis_leaky, self.discriminator_dropout)
        discriminator.cuda()
        return discriminator


    def find_closet_imgs(self,update_imgs, gen_imgs):
        min_dist = torch.zeros(gen_imgs.shape).to(self.device)

        # calc MSE to each update image
        for i in range(gen_imgs.shape[0]):
            sub = gen_imgs - update_imgs[i][0].to(self.device)
            dist = torch.norm(sub.view(self.update_set_size, -1), dim=1).pow(2)
            min_dist[i] = gen_imgs[torch.argmin(dist)]

        return min_dist

    def list_update_sets(self, update_sets):
        listed_update_sets=[None]*self.num_train_update_samples
        for i in range(self.num_train_update_samples):
            listed_update_sets[i]=torch.zeros((self.update_set_size,) + self.image_size)
            for j,set in enumerate(update_sets[i]):
                listed_update_sets[i][j]=set

        return listed_update_sets

    def get_gan_models(self, pretrain_switch):
        if pretrain_switch=="True":
            encoder=torch.load(self.encoder_path, map_location=self.device)
            generator=torch.load(self.generator_path, map_location=self.device)
            discriminator=torch.load(self.discriminator_path, map_location=self.device)


        else:
            encoder = self.init_encoder()

            if self.generator_type=='old':
                generator = self.init_old_generator()
            else:
                generator = self.init_new_generator()


            if self.discriminator_type=='old':
                discriminator = self.init_old_discriminator()
            else:
                discriminator = self.init_new_discriminator()

        return (encoder,generator,discriminator)

# z = torch.FloatTensor(np.random.normal(0, 1, (self.update_set_size, self.noise_dim))).to(self.device)
