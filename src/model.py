import torch
import numpy as np
import pandas as pd
from torch import nn as nn


# VAE hyper-parameters
vae_params = {'encoder':0, 'decoder':0}



class VAE(nn.Module):

    def __init__(self, **kwargs):
        self.encoder = kwargs['encoder']
        self.decoder = kwargs['decoder']

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        z = mu + epsilon*logvar
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z.view(-1, 1, 10))

        return out, mu, logvar, z



def main():
    pass




if __name__ == '__main__':
    main()
