# GANS
This project is inspired by the *Huggingface*. While working on projects unrelated to NLP, I really miss great documentation and well thought API that Hugginface contirbuted to the NLP echosystem. I therefore decided to start the project focused on imaged. The overall idea is to collect (if exist) or to create a a bunch of image models with easy to use APIs and decent documentation.  Each model is an installable Python package.  All the code is in PyTorch.  
## Sequence GAN
This is a very basic GAN model based on fully connected network.  This is not something anyone would ever use in producting yet it is pretty useful if for building an overall understanding of generative adversarial network.  The model demonstrates decent performance on MNIST dataset (28x28 pixels images). Using GPU is highly recommended, minimum 100 iterations are suggested
### User Manual (work in progress)
**Model Package** - gan_dnn. 
**Model Class**  - GAN_MLP(). 
*Example*. 
   from gan_dnn import model. 
   example_model = model.GAN_MLP(). 
