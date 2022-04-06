from operator import neg
import torch
import numpy as np
import pandas as pd
from torch import nn as nn



<<<<<<< HEAD
# PLS hyper-parameters
# cnv = [in_channels, out_chnnels, kernel_size, strids]
pls_conv_params = {'cnv1':[1, 8, 7, 5], 'cnv2':[8, 16, 5, 3], 'cnv3':[16, 32, 4, 2], 'output_dim':1, 'input_dim':1776}
pls_linear_params = {'input_dim': 10, 'output_dim': 1, 'h1':8, 'h2': 5}

class PLS(nn.Module):

    def __init__(self, **kwargs):
        super(PLS, self).__init__()
        self.blk1 = self._block(inCh=kwargs['cnv1'][0], outCh=kwargs['cnv1'][1], krSz=kwargs['cnv1'][2], stride=kwargs['cnv1'][3])
        self.blk2 = self._block(inCh=kwargs['cnv2'][0], outCh=kwargs['cnv2'][1], krSz=kwargs['cnv2'][2], stride=kwargs['cnv2'][3])
        self.blk3 = self._block(inCh=kwargs['cnv3'][0], outCh=kwargs['cnv3'][1], krSz=kwargs['cnv3'][2], stride=kwargs['cnv3'][3])
       
        self.gap = nn.AvgPool1d(kernel_size=19, stride=19)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(in_features=32*3, out_features=kwargs['output_dim'])
        self.sigmoid = nn.Sigmoid()


    def _block(self, inCh, outCh, krSz, stride, pdd=0):
        return nn.Sequential(
            nn.Conv1d(in_channels=inCh, out_channels=outCh, kernel_size=krSz, stride=stride, padding=pdd),
            nn.BatchNorm1d(outCh),
            nn.LeakyReLU(negative_slope=0.1)
        )
    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.out(x)
        out = self.sigmoid(x)

        return out

def main():
    x = torch.Tensor(torch.randn(40, 1, 1776))
    pls = PLS(**pls_conv_params)
    print(pls)
    out = pls(x)
    print(out.size())
=======
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
    x = torch.Tensor(torch.randn(100, 1, 2000))
    encoder = Encoder(**encoder_linear_params)
    mu, logvar = encoder(x)
    print(f"shape(x) = {x.size()}")
    print(f"shape(mu) = {mu.size()}")
    print(f"shape(logvar) = {logvar.size()}")
    print(encoder)
>>>>>>> model/Encoder




if __name__ == '__main__':
    main()
