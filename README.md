# Denoising Diffusion Probabilistic Model

An implementation of DDPM for the MNIST dataset.<p>

This is based on the following paper:<br>
https://arxiv.org/pdf/2006.11239

# Model Components

- **`model/model.py`**: Contains the implementation of the DDPM model.
- **`model/blocks.py`**: Contains the building blocks used in the DDPM model, `DownSampleBlock`, `IdentityBlock`, and `UpSampleBlock`.
- **`utils/dataloader.py`**: Contains the function to get the data loaders for the MNIST dataset.
- **`utils/noise_scheduler.py`**: Contains the implementation of the `NoiseScheduler` class which implements a linear scheduler for noise.
- **`train.py`**: Script to train the model and save it.
