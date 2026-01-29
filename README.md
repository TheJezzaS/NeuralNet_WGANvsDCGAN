# DCGAN and WGAN-GP on Fashion-MNIST

NOTE:
first clone the github repo via the first code block!
then you can run the implemented code.
link: https://github.com/TheJezzaS/NeuralNet_WGANvsDCGAN.git

This project implements **DCGAN** and **WGAN-GP** using PyTorch and trains them on the **Fashion-MNIST** dataset.  
The architecture is adapted from the *simple CIFAR-10 model* described in  
**“Improved Training of Wasserstein GANs” (Gulrajani et al.)**, modified for 1×28×28 grayscale images.

The code supports:
- Training DCGAN and WGAN-GP via command-line arguments
- Saving trained model weights
- Easy generation of new images using trained generators



## Use:
### to train DCGAN:
python train.py --mode dcgan

###  Train WGAN-GP
python train.py --mode wgan


Optional flags:
--epochs, default=20
--batch_size, default=64
--lr, default=1e-4
--z_dim, default=128
--n_critic, default=5

After training, the following files are saved automatically:
checkpoints/G_dcgan.pth
checkpoints/D_dcgan.pth
checkpoints/G_wgan.pth
checkpoints/D_wgan.pth


## Generating New Images

New images can be generated without retraining by loading a trained generator checkpoint.

### Generate images with DCGAN
python generate.py --checkpoint checkpoints/G_dcgan.pth

### Generate images with WGAN-GP
python generate.py --checkpoint checkpoints/G_wgan.pth

Optional flags:
--z_dim, default=128
--n_samples, default=16
"--show_real"