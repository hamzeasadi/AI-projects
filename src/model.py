import torch
import numpy as np
import pandas as pd
from torch import nn as nn



# PLS hyper-parameters
# cnv = [in_channels, out_chnnels, kernel_size, strids]
pls_conv_params = {'cnv1':[1, 8, 7, 5], 'cnv2':[8, 32, 5, 3], 'cnv3':[32, 64, 4, 2], 'output_dim':1, 'input_dim':600}
pls_linear_params = {'input_dim': 10, 'output_dim': 1, 'h1':8, 'h2': 5}

class PLS(nn.Module):

    def __init__(self, **kwargs):
        super(PLS, self).__init__()
        self.blk1 = self._block(inCh=kwargs['cnv1'][0], outCh=kwargs['cnv1'][1], krSz=kwargs['cnv1'][2], stride=kwargs['cnv1'][3])
        self.blk2 = self._block(inCh=kwargs['cnv2'][0], outCh=kwargs['cnv2'][1], krSz=kwargs['cnv2'][2], stride=kwargs['cnv2'][3])
        self.blk3 = self._block(inCh=kwargs['cnv3'][0], outCh=kwargs['cnv3'][1], krSz=kwargs['cnv3'][2], stride=kwargs['cnv3'][3])
       
        self.gap = nn.AvgPool1d(kernel_size=18, stride=1)
        self.flatten = nn.Flatten()
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
        out = self.sigmoid(x)

        return out

def main():
    x = torch.Tensor(torch.randn(40, 1, 600))
    pls = PLS(**pls_conv_params)
    print(pls)
    out = pls(x)
    print(out.size())




if __name__ == '__main__':
    main()
