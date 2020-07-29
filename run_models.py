from configuration.config import Configuration
from utils import Utils
import numpy as np
from data_extractor import DataExtractor
from training_models.train_gan import TrainGan
from training_models.train_shadow_model import TrainShadow
from training_models.train_update_model import TrainUpdateModels
from image_generator import ImageGenerator
from training_models.base_train import BaseTrain
from plotter import plot_dict
from utils import Utils
import torch

class run_model():
    def __init__(self):
        self.config=Configuration().get_config()
        self.experiment_parameters_list=Configuration().get_experiments_list()
        self.device=Utils().get_device()

    def run(self):
        np.random.seed(self.config['seed'])

        image_dict = {}
        for i,exppriment_parameters in enumerate(self.experiment_parameters_list):
            print(exppriment_parameters)
            (train_data, probe_data, train_update_sets, test_update_sets,  test_set)=DataExtractor().get_data()
            shadow_model=TrainShadow(exppriment_parameters).get_trained_model(train_data, test_set)
            train_deltas=TrainUpdateModels('train',exppriment_parameters).get_update_dataset(probe_data, train_update_sets, shadow_model)
            test_deltas=TrainUpdateModels('test',exppriment_parameters).get_update_dataset(probe_data, test_update_sets, shadow_model)
            (generator, encoder)=TrainGan(train_deltas,train_update_sets, exppriment_parameters).get_GAN()
            generated_images=ImageGenerator(exppriment_parameters).get_images(test_update_sets,test_deltas, encoder, generator)
            Utils().add_images_to_dict(3,image_dict,generated_images, exppriment_parameters)
            print (f'finished running experiment {i+1} out of {len(self.experiment_parameters_list)}')

        plot_dict(image_dict, len(self.experiment_parameters_list))
        Utils().plot_update(test_update_sets[1])
        Utils().plot_generated(generated_images[1])

        return generated_images


if __name__ == '__main__':
    print(run_model().run())