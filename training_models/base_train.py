from configuration.config import Configuration
from utils import Utils
from datetime import datetime

class BaseTrain():
    def __init__(self,parameters):
        self.config=Configuration().get_config()
        self.device=Utils().get_device()
        self.log_dir='runs'
        self.model_dir='saved_models/'
        self.update_set_size=self.config['models']['sample_sizes']['update_size']
        self.num_train_update_samples=self.config['models']['sample_sizes']['num_train_update_samples']
        self.num_test_update_samples = self.config['models']['sample_sizes']['num_test_update_samples']
        self.image_size=(3,32,32)
        self.time = datetime.now().strftime("%d-%m-%H-%M-%S")
        self.noise_dim=self.config['models']['GAN']['noise_dim']
        self.datasets_directory = f'data/datasets'
        self.shadow_num_epochs=self.config['models']['shadow_model']['num_epochs']


        self.shadow_in_channel=parameters['shdow_in_channel']
        self.shadow_out_channels =parameters['shdow_out_channels']
        self.shadow_fc_size =self.shadow_out_channels[1]*25
        self.shadow_learning_rate=parameters['shdow_learning_rate']

        self.update_learning_rate=parameters['update_learning_rate']
        self.update_shadow_epochs=parameters['update_shadow_epochs']

        self.encoder_leky_reul=parameters['encoder_leaky_relu']
        self.encoder_dropout=parameters['encoder_dropout']
        self.dis_leaky=parameters['dis_leaky']
        self.dis_ndf=parameters['dis_ndf']
        self.generator_leaky=parameters['generator_leaky']
        self.generator_ngf=parameters['generator_ngf']
        self.gan_loss_weight=parameters['gan_loss_weight']
        self.discriminator_type=parameters['discriminator_type']
        self.generator_type=parameters['generator_type']
        self.noisy_label_change=parameters['noisy_label_change']
        self.discriminator_dropout=parameters['discriminator_dropout']
        self.lr_gamma=parameters['lr_gamma']

        self.cache_name='cache/'+ Utils.get_folder_name(self.update_set_size, self.num_train_update_samples,self.num_test_update_samples)
    def add_dimension_to_mu(self, mu):
        mu=mu.unsqueeze(2)
        mu=mu.unsqueeze(3)
        return mu
