import torch
from torch import nn


def generator_block(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))


def discriminator_block(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.LeakyReLU(0.2, inplace=True))


class Generator(nn.Module):
    # The generator class takes 3 arguments (noise_vector_dim, expanded_image_dim, hidden_layer_dim)
    def __init__(self, noise_vector_dim=5, expanded_image_dim=784, hidden_layer_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(generator_block(noise_vector_dim, hidden_layer_dim),
                                 generator_block(hidden_layer_dim, hidden_layer_dim * 2),
                                 generator_block(hidden_layer_dim * 2, hidden_layer_dim * 3),
                                 nn.Linear(hidden_layer_dim * 3, expanded_image_dim),
                                 nn.Sigmoid())

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


class Discriminator(nn.Module):
    def __init__(self, expanded_image_dim=784, hidden_layer_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(discriminator_block(expanded_image_dim, hidden_layer_dim * 3),
                                  discriminator_block(hidden_layer_dim * 3, hidden_layer_dim),
                                  nn.Linear(hidden_layer_dim, 1))

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc
