import torch

from gan import GAN
from generator import Generator

import numpy as np
from matplotlib import pyplot as plt, gridspec

PATH_TO_MODEL_STATE = "saved_models/run_1/by_epoch/090_checkpoint"
N_SAMPLES = 10

def generate_samples(
    generator : Generator,
    device,
    n_samples: int = 10,
):
    z = torch.randn(n_samples, 1, 100, device=device)
    generated_imgs = generator(z)
    
    return generated_imgs

def drawing_imgs(imgs, saved_path='generation/generated/samples.png'):
    nrow = 3
    ncol = 8
    fig = plt.figure(figsize=((ncol+1)*2, (nrow+1)*2)) 
    fig.suptitle('Generated images', fontsize=30)
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    stop = False
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            if idx >= len(imgs):
                stop = True
                break
            img = np.reshape(imgs[idx], (28, 28))
            ax = plt.subplot(gs[i,j])
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        if stop:
            break
        
    # plt.show()
    fig.savefig(saved_path)
    
    
if __name__ == "__main__":
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': module.state_dict(),
                'list_optimizer_state_dict': [optim.state_dict() for optim in optim_lst],
                'loss': disc_train_loss + gen_train_loss,
                },
                "{}/{:03}_checkpoint".format(checkpoint_dir, epoch)
            )
    """
    device = torch.device("mps")
    model = GAN(device=device)
    checkpoint = torch.load(PATH_TO_MODEL_STATE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    imgs = generate_samples(model.generator, device, N_SAMPLES)

    imgs = [i.detach().cpu().numpy() for i in imgs]
    # print(imgs)
    
    drawing_imgs(imgs)
