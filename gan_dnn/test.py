import torch

from gan_dnn import ns
from gan_dnn import model

print(type(ns.get_noise))
#Create a network
my_gan = model.GAN_MLP()
print(type(my_gan))