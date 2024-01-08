import torch.nn as nn

__all__ = ['Generator']

class Generator(nn.Module):
  '''
  Generator class. Accepts a tensor of size 100 as input as outputs another
  tensor of size 784. Objective is to generate an output tensor that is
  indistinguishable from the real MNIST digits 
  '''
  
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Sequential(nn.Linear(in_features=100, out_features=256),
                                nn.LeakyReLU())
    self.layer2 = nn.Sequential(nn.Linear(in_features=256, out_features=512),
                                nn.LeakyReLU())
    self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=1024),
                                nn.LeakyReLU())
    self.output = nn.Sequential(nn.Linear(in_features=1024, out_features=28*28),
                                nn.Tanh())

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.output(x)
    return x