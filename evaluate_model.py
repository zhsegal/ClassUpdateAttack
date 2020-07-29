import torch
from training_models.base_train import BaseTrain

class EvaluateModel():
    def __init__(self):
        pass


    def evaluate(self, disc_loss, mse_loss, dis_loss,  epoch, writer):


        writer.add_scalar('discriminator training loss',disc_loss,epoch)

        writer.add_scalar('mse training loss', mse_loss,epoch)

        writer.add_scalar('discrminator_g training loss', dis_loss, epoch)
