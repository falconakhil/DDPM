import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoiseScheduler:
  def __init__(self,beta_0=0.0001,beta_T=0.02,T=1000):
    self.beta_0=beta_0
    self.beta_T=beta_T
    self.T=T
    self.sqrt_alpha_bar=[1]
    self.sqrt_one_minus_alpha_bar=[0]
    self.alpha_bar=[1]
    self.alpha=[1]

    alpha_bar=1
    for i in range(0,T):
      beta=self.beta_0+(self.beta_T-self.beta_0)*i/T
      alpha=1-beta
      alpha_bar*=alpha
      self.alpha.append(alpha)
      self.alpha_bar.append(alpha_bar)
      self.sqrt_alpha_bar.append(alpha_bar**0.5)
      self.sqrt_one_minus_alpha_bar.append((1-alpha_bar)**0.5)

    self.sqrt_alpha_bar=torch.Tensor(self.sqrt_alpha_bar).to(device)
    self.sqrt_one_minus_alpha_bar=torch.Tensor(self.sqrt_one_minus_alpha_bar).to(device)
    self.alpha_bar=torch.Tensor(self.alpha_bar).to(device)
    self.alpha=torch.Tensor(self.alpha).to(device)

  def get_timestep(self,x_0,t):
    sqrt_alpha_bar_t=self.sqrt_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    epsilon=torch.randn_like(x_0,)
    return sqrt_alpha_bar_t*x_0 + epsilon*sqrt_one_minus_alpha_bar_t,epsilon

  def sample_X_0(self,X_t,epsilon,t):
    sqrt_alpha_bar_t=self.sqrt_alpha_bar[t]
    sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t]
    X_0=(X_t-sqrt_one_minus_alpha_bar_t*epsilon)/sqrt_alpha_bar_t
    X_0=torch.clamp(X_0,-1,1)
    return X_0

  def sample_previous_timestep(self,X_t,epsilon,t):
    sqrt_one_minus_alpha_bar_t=self.sqrt_one_minus_alpha_bar[t]

    alpha=self.alpha[t]
    beta=1-alpha

    X_t1=(X_t-(beta*epsilon)/sqrt_one_minus_alpha_bar_t)/(alpha**0.5)
    if t!=1:
      var=beta*(1-self.alpha_bar[t-1])/(1-self.alpha_bar[t])
      z=torch.randn_like(X_t1).to(device)
      X_t1+=z*(var**0.5)

    return X_t1
