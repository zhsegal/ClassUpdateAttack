import torch
from training_models.base_train import BaseTrain
import matplotlib.pyplot as plt
import os
import cv2
from scipy.cluster.vq import vq
from utils import Utils

class ImageGenerator(BaseTrain):
    def __init__(self, parameters):
        super(ImageGenerator, self).__init__(parameters=parameters)
        self.parameters = self.config['image_generator']
        self.img_generator_epohcs=self.parameters['epochs']
        self.image_folder= 'images_' + Utils().get_folder_name(list(parameters.values()))
        self.image_file_name='images.pt'
        self.update_imgs_file_name='update_images.pt'
        self.image_directory=self.cache_name + self.image_folder
        self.image_path= self.image_directory + self.image_file_name
        self.update_image_path = self.image_directory + self.update_imgs_file_name

    def get_images(self,test_update_sets, deltas, encoder, generator ):
        if os.path.exists(self.image_path):
            print ('images already exists, loading')
            return torch.load(self.image_path, map_location=torch.device(self.device))
        else:
            print ('images doesnt exists, generating')
            return self.image_generator(test_update_sets, deltas, encoder, generator)

    def image_generator(self,test_update_sets, deltas, encoder, generator):

        final_images=[]
        for i, update_set in enumerate(test_update_sets):
            images_bank=torch.zeros(0, *self.image_size).to(self.device)
            for epoch in range(self.img_generator_epohcs):
                delta=torch.Tensor(deltas[i]).to(self.device)

                z = torch.randn(self.update_set_size, self.noise_dim, 1, 1, device=self.device)
                mu = encoder(delta)
                mu = self.add_dimension_to_mu(mu)

                generated_image = generator(mu.to(self.device), z)
                images_bank = torch.cat((images_bank, generated_image), dim=0)

            center_images=self.cluster(images_bank)
            final_images.append(center_images)
            if i%10==0: print (f'finished update set {i}')


        if os.path.isdir(self.image_directory)==False: os.mkdir(self.image_directory)

        torch.save(final_images, self.image_path)
        torch.save(test_update_sets, self.update_image_path)
        return final_images

    def cluster(self, images_bank):
        Z = images_bank.reshape((images_bank.shape[0], -1))
        Z = Z.cpu().detach().numpy()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = self.update_set_size
        dist, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        closest, distances = vq(center, Z)
        center_images=Z[closest]
        center_images = center_images.reshape(self.update_set_size, *self.image_size)
        center_images=torch.Tensor(center_images)
        return center_images


#Utils().plot_update(update_set)
#Utils().plot_generated(generated_image)