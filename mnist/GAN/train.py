import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import pickle
import os

from gan import GAN
tb_logger = SummaryWriter()


from constant import TRESHOLD, HALF_TRESHOLD
# import dill as pickle


SAVED_DIR = 'saved_models'
RUNNING_NAME = 'run_1'
FINAL_RUN_DIR = "{}/{}".format(SAVED_DIR, RUNNING_NAME)

SAVE_MODEL_EVERY_N_EPOCH = 15

def flatten_image(x):
        return x.view(-1, 784)

def get_loader() -> DataLoader:

    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
        transforms.Lambda(flatten_image)
    ])

    dataset = datasets.MNIST(root='./tmp/MNIST',
                             download=True,
                             transform=mnist_transforms)

    mnist_dataloader = DataLoader(dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=4,
                                  )
    
    return mnist_dataloader

def train(
    module: GAN,
    data_loader,
    max_epoch = 100,
    device = None
):
    assert device is not None
    global tb_logger
    
    optim_lst, _ = module.configure_optimizers()
    module.train()
    for epoch in range(max_epoch):
        print("{}EPOCH={}{}".format(
            HALF_TRESHOLD,
            epoch,
            HALF_TRESHOLD
        ))
        gen_train_loss = 0.
        disc_train_loss = 0.
        for batch_id, batch in enumerate(data_loader):
            X, y = batch
            X = torch.as_tensor(X, device=device)
            batch = [X, y]
            # train generator
            cur_optimizer_idx = 0
            optim_lst[cur_optimizer_idx].zero_grad()
            
            loss = module.training_step(
                batch,
                batch_idx=None,
                optimizer_idx=cur_optimizer_idx
            )
            loss.backward()
            optim_lst[cur_optimizer_idx].step()
            gen_train_loss += loss.item()
            
            # train discriminator
            cur_optimizer_idx = 1
            optim_lst[cur_optimizer_idx].zero_grad()
            loss = module.training_step(
                batch,
                batch_idx=None,
                optimizer_idx=cur_optimizer_idx
            )
            loss.backward()
            optim_lst[cur_optimizer_idx].step()
            disc_train_loss += loss.item()
        
        tb_logger.add_scalar("Loss/train", disc_train_loss + gen_train_loss, epoch)
        tb_logger.add_scalar("Generator loss/train", gen_train_loss, epoch)
        tb_logger.add_scalar("Discriminator loss/train", disc_train_loss, epoch)
        
        tb_logger.flush()
        
        module.training_epoch_end()
        
        print("{}train_loss={}{}".format(
            HALF_TRESHOLD, disc_train_loss + gen_train_loss, HALF_TRESHOLD
        ))

        if epoch % SAVE_MODEL_EVERY_N_EPOCH == 0:
            checkpoint_dir = "{}/by_epoch".format(
                FINAL_RUN_DIR
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': module.state_dict(),
                'list_optimizer_state_dict': [optim.state_dict() for optim in optim_lst],
                'loss': disc_train_loss + gen_train_loss,
                },
                "{}/{:03}_checkpoint".format(checkpoint_dir, epoch)
            )
            with open("{}/test_progression.pickle".format(FINAL_RUN_DIR), 'wb+') as f:
                pickle.dump(module.test_progression, f)
            
    
    tb_logger.close()

    
    
        
    

if __name__ == "__main__":
    mnist_dataloader = get_loader()
    
    # TODO: add cpu, gpu
    device = torch.device("mps")
    MAX_EPOCH = 100
    module = GAN(device=device)
    module.to(device)
    train(
        module=module,
        data_loader=mnist_dataloader,
        max_epoch=MAX_EPOCH,
        device=device
    )
    
    

    
    
    
    
    
