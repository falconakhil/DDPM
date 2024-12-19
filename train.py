from model.model import DDPM
from utils.noise_scheduler import NoiseScheduler
from utils.dataloader import get_dataloader

import torch

from tqdm import tqdm

def train(model,train_loader,test_loader,loss_fn,optimizer,ns,epochs,loss_history,val_loss_history):

  for epoch in range(1,epochs+1):
    print(f"Epoch [{epoch}/{epochs}]:")
    # Train
    model.train()
    avg_loss=0
    for batch_idx,(X,y) in enumerate(tqdm(train_loader,desc="Progress")):
      optimizer.zero_grad()
      X_0=X.to(device)
      t=torch.randint(0,ns.T,(X_0.shape[0],))

      X_t,real_noise=ns.get_timestep(X_0,t)
      pred_noise=model(X_t,t)
      loss=loss_fn(pred_noise,real_noise)

      loss.backward()
      optimizer.step()

      avg_loss+=loss.item()

    avg_loss/=len(train_loader)
    loss_history.append(avg_loss)

    # Evaluate
    model.eval()
    avg_val_loss=0
    with torch.no_grad():
      for batch_idx,(X,y) in enumerate(test_loader):
        X_0=X.to(device)
        t=torch.randint(1,ns.T,(X_0.shape[0],))

        X_t,real_noise=ns.get_timestep(X_0,t)
        pred_noise=model(X_t,t)

        loss=loss_fn(pred_noise,real_noise)
        avg_val_loss+=loss.item()

      avg_val_loss/=len(test_loader)
      val_loss_history.append(avg_val_loss)

    # Display progress
    print(f"loss={avg_loss} val_loss={avg_val_loss}]")

if __name__=='__main__':

    #Config
    epochs=20
    latent_dim=256
    learning_rate=1e-3

    # Get the dataloaders
    train_loader,test_loader=get_dataloader(batch_size=64)

    # Use GPU if available
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the noise scheduler
    ns=NoiseScheduler()

    # Initialize the model
    model=DDPM(latent_dim=latent_dim).to(device)

    # Loss function and optimizer
    loss_fn=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

    # Loss History
    loss_history=[]
    val_loss_history=[]

    # Train the model
    train(model,train_loader,test_loader,loss_fn,optimizer,ns,epochs,loss_history,val_loss_history)

    # Save the model
    torch.save(model.state_dict(),'model.torch')
