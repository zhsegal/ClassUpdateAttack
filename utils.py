from configuration.config import Configuration
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import random

class Utils():
    def __init__(self):
        self.config=Configuration().get_config()
        self.device=torch.device(self.config['device']['gpu'] if torch.cuda.is_available() else self.config['device']['cpu'])

    def get_model_name(self, *args):
        name='_'.join(str(value) for value in args) + '.pt'

        return name

    def get_device(self):
        return torch.device(self.config['device']['gpu'] if torch.cuda.is_available() else self.config['device']['cpu'])

    def matplotlib_imshow(self,img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def plot_update(self, real_batch):
        dataloader = torch.utils.data.DataLoader(real_batch, batch_size=10)
        real_batch = next(iter(dataloader))

        plt.figure(figsize=(2, 5))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:10], padding=2, normalize=True).cpu(),
                         (1, 2, 0)))

        plt.show()
        return 'ploted'

    def plot_generated(self, generated_image):
        plt.figure(figsize=(2, 5))
        plt.axis("off")
        plt.title("generated Images")
        plt.imshow(
            np.transpose(vutils.make_grid(generated_image.detach().cpu()[:10], padding=2, normalize=True).cpu(),
                         (1, 2, 0)))

        return 'ploted'

    @classmethod
    def get_folder_name(self,*args):
        folder_name='_'.join(str(value) for value in args) + '/'
        return folder_name

    def add_images_to_dict(self, num_of_images, image_dict, generated_images,exppriment_parameters):
        name = Utils().get_folder_name(list(exppriment_parameters.values()))
        for i in range(num_of_images):
            random_index=random.randint(0,len(generated_images)-1)
            image_dict[name + f'_{i}']=generated_images[random_index]
        return image_dict