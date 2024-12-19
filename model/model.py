import torch
from torch import nn

from blocks import DownSampleBlock, IdentityBlock, UpSampleBlock

import math

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPM(nn.Module):
  def __init__(self,latent_dim=256,embedding_n=1000):
    super(DDPM,self).__init__()

    # Hyperparams
    self.latent_dim=latent_dim
    self.embedding_n=embedding_n

    # Encoder
    self.down1=DownSampleBlock(in_channels=1,out_channels=32)
    self.did1=IdentityBlock(channels=32)

    self.down2=DownSampleBlock(in_channels=32,out_channels=64)
    self.did2=IdentityBlock(channels=64)

    self.down3=DownSampleBlock(in_channels=64,out_channels=128)
    self.did3=IdentityBlock(channels=128)

    self.down4=DownSampleBlock(in_channels=128,out_channels=256)
    self.did4=IdentityBlock(channels=256)

    self.flatten=nn.Flatten()
    self.to_latent=nn.Linear(in_features=1024,out_features=self.latent_dim)
    self.latent_relu=nn.ReLU()

    # Timestep Embedding Network
    self.embed_silu=nn.SiLU()
    self.embed_fc=nn.Linear(in_features=self.latent_dim,out_features=64*8*8)

    # Decoder
    self.from_linear=nn.Linear(in_features=self.latent_dim,out_features=1024)
    self.unflatten=nn.Unflatten(dim=1,unflattened_size=(256,2,2))
    self.base_relu=nn.ReLU()

    self.up1=UpSampleBlock(in_channels=256,out_channels=128)
    self.uid1=IdentityBlock(channels=128,norm=False)

    self.up2=UpSampleBlock(in_channels=128,out_channels=64)
    self.uid2=IdentityBlock(channels=64,norm=False)

    self.up3=UpSampleBlock(in_channels=64,out_channels=32)
    self.uid3=IdentityBlock(channels=32,norm=False)

    self.up4=UpSampleBlock(in_channels=32,out_channels=1,final=True)
    self.uid4=IdentityBlock(channels=1,final=True,norm=False)


  def encode_t(self,t=None):

    t=t.to(device).unsqueeze(1).float()
    i=torch.arange(0,self.latent_dim,2).to(device)
    div_term=torch.exp(-2*i/self.latent_dim*math.log(self.embedding_n)).unsqueeze(0)
    encoding=torch.zeros((t.shape[0],self.latent_dim)).to(device)
    encoding[:,0::2]=torch.sin(torch.matmul(t,div_term)).to(device)
    encoding[:,1::2]=torch.cos(torch.matmul(t,div_term)).to(device)

    return encoding

  def forward(self,x,t):

    # Create timestep embedding
    t_encoding=self.encode_t(t)
    t_embedding=self.embed_fc(t_encoding)
    t_embedding=self.embed_silu(t_embedding)
    t_embedding=t_embedding.view(-1,64,8,8)

    #Encoder
    down1=self.down1(x)
    did1=self.did1(down1)

    down2=self.down2(did1)
    did2=self.did2(down2)
    did2=did2+t_embedding  # Condition layer with timestep

    down3=self.down3(did2)
    did3=self.did3(down3)

    down4=self.down4(did3)
    did4=self.did4(down4)

    flatten=self.flatten(did4)
    latent=self.to_latent(flatten)
    latent=self.latent_relu(latent)

    # Decoder
    from_linear=self.from_linear(latent)
    unflatten=self.unflatten(from_linear)
    base_relu=self.base_relu(unflatten)

    up1=self.up1(base_relu)
    uid1=self.uid1(up1)
    uid1=uid1+did3

    up2=self.up2(uid1)
    uid2=self.uid2(up2)
    uid2=uid2+did2

    up3=self.up3(uid2)
    uid3=self.uid3(up3)
    uid3=uid3+did1

    up4=self.up4(uid3)
    uid4=self.uid4(up4)

    return uid4


  def sample(self,X_T,ns):
    X_t=X_T
    for t in range(ns.T,0,-1):
      with torch.no_grad():
        epsilon=self.forward(X_t,torch.Tensor([t]).to(device))
      X_t=ns.sample_previous_timestep(X_t,epsilon,t)
    return X_t
