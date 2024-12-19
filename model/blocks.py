import torch
from torch import nn

class DownSampleBlock(nn.Module):
    '''
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ├─Sequential: 1-1                        [-1, 32, 16, 16]          --
    |    └─Conv2d: 2-1                       [-1, 32, 16, 16]          896
    |    └─GroupNorm: 2-2                    [-1, 32, 16, 16]          64
    |    └─ReLU: 2-3                         [-1, 32, 16, 16]          --
    |    └─Conv2d: 2-4                       [-1, 32, 16, 16]          9,248
    ├─Conv2d: 1-2                            [-1, 32, 16, 16]          128
    ├─GroupNorm: 1-3                         [-1, 32, 16, 16]          64
    ├─ReLU: 1-4                              [-1, 32, 16, 16]          --
    ==========================================================================================
    Total params: 10,400
    Trainable params: 10,400
    Non-trainable params: 0
    Total mult-adds (M): 2.62
    ==========================================================================================
    Input size (MB): 0.01
    Forward/backward pass size (MB): 0.31
    Params size (MB): 0.04
    Estimated Total Size (MB): 0.36
    ==========================================================================================
    '''

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        )
        self.res_conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=2)
        self.out_bn=nn.GroupNorm(8,out_channels)
        self.out_relu=nn.ReLU()

    def forward(self,x):
        out=self.block(x)+self.res_conv(x)
        out=self.out_bn(out)
        out=self.out_relu(out)
        return out



class IdentityBlock(nn.Module):
    '''
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ├─Sequential: 1-1                        [-1, 8, 32, 32]           --
    |    └─Conv2d: 2-1                       [-1, 8, 32, 32]           584
    |    └─GroupNorm: 2-2                    [-1, 8, 32, 32]           16
    |    └─ReLU: 2-3                         [-1, 8, 32, 32]           --
    |    └─Conv2d: 2-4                       [-1, 8, 32, 32]           584
    ├─GroupNorm: 1-2                         [-1, 8, 32, 32]           16
    ├─ReLU: 1-3                              [-1, 8, 32, 32]           --
    ==========================================================================================
    Total params: 1,200
    Trainable params: 1,200
    Non-trainable params: 0
    Total mult-adds (M): 1.18
    ==========================================================================================
    Input size (MB): 0.03
    Forward/backward pass size (MB): 0.25
    Params size (MB): 0.00
    Estimated Total Size (MB): 0.29
    ==========================================================================================
    '''

    def __init__(self,channels,final=False,norm=True):
        super().__init__()

        self.block=nn.Sequential(
        nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1),
        nn.GroupNorm(8,channels) if norm and not final else nn.Identity(),
        nn.ReLU(),
        nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1)
        )

        self.out_bn=nn.GroupNorm(8,channels) if norm and not final else nn.Identity()
        if final:
            self.out_act=nn.Identity()
        else:
            self.out_act=nn.ReLU()
    def forward(self,x):
        out=self.block(x)+x
        out=self.out_bn(out)
        out=self.out_act(out)
        return out


class UpSampleBlock(nn.Module):
    '''
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    ├─Sequential: 1-1                        [-1, 3, 64, 64]           --
    |    └─ConvTranspose2d: 2-1              [-1, 3, 64, 64]           867
    |    └─ReLU: 2-2                         [-1, 3, 64, 64]           --
    |    └─Conv2d: 2-3                       [-1, 3, 64, 64]           84
    ├─ConvTranspose2d: 1-2                   [-1, 3, 64, 64]           387
    ├─ReLU: 1-3                              [-1, 3, 64, 64]           --
    ==========================================================================================
    Total params: 1,338
    Trainable params: 1,338
    Non-trainable params: 0
    Total mult-adds (M): 5.44
    ==========================================================================================
    Input size (MB): 0.12
    Forward/backward pass size (MB): 0.28
    Params size (MB): 0.01
    Estimated Total Size (MB): 0.41
    ==========================================================================================
    '''

    def __init__(self,in_channels,out_channels,final=False):
        super().__init__()
        self.block=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
        )

        self.res_conv=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2,padding=0,output_padding=0)
        self.out_relu=nn.ReLU()

    def forward(self,x):
        out=self.block(x)+self.res_conv(x)
        out=self.out_relu(out)
        return out
