from operator import neg
import torch
import numpy as np
import pandas as pd
from torch import nn as nn



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


# model hyper
decoder_linear_params = {
    'h1': 100, 'h2': 500, 'output_dim':2000, 'latent_dim': 10
}

class Decoder(nn.Module):
    
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.blk1 = self._block(inFeat=kwargs['latent_dim'], outFeat=kwargs['h1'])
        self.blk2 = self._block(inFeat=kwargs['h1'], outFeat=kwargs['h2'])
        self.out = nn.Linear(in_features=kwargs['h2'], out_features=kwargs['output_dim'])
        self.sigmoid = nn.Sigmoid()

    def _block(self, inFeat, outFeat):
        return nn.Sequential(
            nn.Linear(in_features=inFeat, out_features=outFeat),
            nn.BatchNorm1d(1), nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        out = self.out(x)

        return out


# VAE hyper-parameters
vae_params = {'encoder':0, 'decoder':0}

class VAE(nn.Module):

    def __init__(self, **kwargs):
        super(VAE, self).__init__()
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
    x = torch.Tensor(torch.randn(100, 1, 2000))

    encoder = Encoder(**encoder_linear_params)
    
    decoder = Decoder(**decoder_linear_params)

    vae_params['encoder'] = encoder
    vae_params['decoder'] = decoder
    vae = VAE(**vae_params)

    print(vae)





if __name__ == '__main__':
    main()
