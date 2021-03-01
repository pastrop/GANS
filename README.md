# GANS
This project is inspired by the *Huggingface*. While working on projects unrelated to NLP, I really miss great documentation and well-thought APIs that Hugginface contirbuted to the NLP echosystem. I therefore decided to start the project focused on image processing. The overall idea is to collect (if exist) or to create a a bunch of image models with easy to use APIs and decent documentation as installable Python packages.  All code is in PyTorch.  
## Sequence GAN
The first model available is a very basic GAN  based on fully connected network architecture.  This is not something anyone would ever use in production yet it is useful if for building an overall understanding of generative adversarial network.  The model demonstrates decent performance on MNIST dataset (28x28 pixels images). Using GPU is highly recommended, minimum 100 epochs are suggested
### User Manual (work in progress)
**Installation from Github(currently from develop branch)** - </br> 
```pip install git+https://github.com/pastrop/GANS.git@develop#egg=GANS```</br>
**Model Package** - GANS</br>
**Model Class**  - GAN_MLP(z_dim = 64, lr = 0.00001,device = 'cpu')</br> 
*z_dim* - noize vector dimensionality</br>
*lr* - learning rate</br>
*device* - 'gpu' or 'tpu' is recommended</br>
*Example: Initialzing*</br> 
```
from GANS import model
example_model = model.GAN_MLP(device = 'gpu')
```
**Methods & Variables**</br> 
example_model.gen_opt - holds the generator gradient (Adam is used)</br>
example_model.desc_opt - holds the discriminator gradient (Adam is used)</br>
example_model.get_disc_loss(batch_size, batch_of_real_images) - calculates discriminator loss
example_model.get_gen_loss(batch_size) - calculates generator loss

*Example: Calling Methods & Using Class Variables*</br> 
```
example_model.gen_opt
example_model.desc_opt
example_model.get_disc_loss(batch_size, batch_of_real_images) 
example_model.get_gen_loss(batch_size)
```


**Dependencies**</br>
*Torch*</br>
*Torchvision & Pyplot* - if you want to have vizualization</br>

   
