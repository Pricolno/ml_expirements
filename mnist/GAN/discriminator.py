import torch.nn as nn


class Discriminator(nn.Module):
  '''
  Discriminator class. Accepts a tensor of size 784 as input and outputs
  a tensor of size 1 as  the predicted class probabilities
  (generated or real data)
  '''

  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(in_features=28*28, out_features=1024),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                nn.Sigmoid())
    
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x