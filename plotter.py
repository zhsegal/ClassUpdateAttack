import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def plot_dict(image_dict, num_experiments):
    fig=plt.figure()
    for i, title in enumerate(image_dict.keys()):
        ax=fig.add_subplot(num_experiments*3//2,2,i+1)
        imgplot=plt.imshow( np.transpose(vutils.make_grid(image_dict[title].detach().cpu()[:10], padding=2, normalize=True).cpu(), (1, 2, 0)))
        ax.set_title(title)
    plt.show()
    print('all done!')
