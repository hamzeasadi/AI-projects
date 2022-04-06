from operator import neg
import torch
import numpy as np
import pandas as pd
from torch import nn as nn



# model hyper
encoder_linear_params = {
    'h1': 500, 'h2': 100, 'input_dim':2000, 'latent_dim': 10
}


class Encoder(nn.Module):
    
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.blk1 = self._block(inFeat=kwargs['input_dim'], outFeat=kwargs['h1'])
        self.blk2 = self._block(inFeat=kwargs['h1'], outFeat=kwargs['h2'])
        self.mu = nn.Linear(in_features=kwargs['h2'], out_features=kwargs['latent_dim'])
        self.logvar = nn.Linear(in_features=kwargs['h2'], out_features=kwargs['latent_dim'])


    def _block(self, inFeat, outFeat):
        return nn.Sequential(
            nn.Linear(in_features=inFeat, out_features=outFeat),
            nn.BatchNorm1d(1), nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar




def main():
    pass




if __name__ == '__main__':
    main()
