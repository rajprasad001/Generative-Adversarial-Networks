import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from Gen_Disc.mlp_gen_disc import Generator, Discriminator
import matplotlib.pyplot as plt

torch.manual_seed(0)


def get_noise(n_samples, noise_vector_dim, device='cpu'):
    return torch.randn(n_samples, noise_vector_dim, device=device)


criterion = nn.BCEWithLogitsLoss()
n_epochs = 100
noise_vector_dim = 10
display_step = 500
batch_size = 128
lr = 0.0001
device = 'cuda'
dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), batch_size=batch_size,
                        shuffle=True)

gen = Generator(noise_vector_dim).to(device)
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator().to(device)
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr)


def generator_loss(gen, disc, criterion, num_images, noise_vector_dim, device):
    noise = get_noise(num_images, noise_vector_dim, device=device)
    fake_image = gen(noise)
    discriminator_fake_pred = disc(fake_image)
    generator_loss = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))
    return fake_image, generator_loss


def discriminator_loss(fake_image, disc, criterion, real_images):
    fake_image = fake_image
    discriminator_fake_pred = disc(fake_image.detach())
    discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))
    discriminator_real_pred = disc(real_images)
    discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))
    disc_loss = (discriminator_fake_loss + discriminator_real_loss) / 2
    return disc_loss


def show_images(image_tensor, num_images=9, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=3)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True

for epoch in range(n_epochs):
    for real_image, _ in tqdm(dataloader):
        current_batch_size = len(real_image)
        real = real_image.view(current_batch_size, -1).to(device)

        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        gen_optimizer.zero_grad()
        fake_image, gen_loss = generator_loss(gen, disc, criterion, current_batch_size, noise_vector_dim, device)
        gen_loss.backward()
        gen_optimizer.step()

        disc_optimizer.zero_grad()
        disc_loss = discriminator_loss(fake_image, disc, criterion, real)
        disc_loss.backward(retain_graph=True)
        disc_optimizer.step()

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        if step % display_step == 0 and step > 0:
            print(f'Epoch : {epoch}, Step : {step}, '
                  f'Generator_loss : {mean_generator_loss}, '
                  f'Discriminator_loss : {mean_discriminator_loss}')

            img = fake_image.size()
            show_images(fake_image)
            show_images(real)
            mean_discriminator_loss = 0
            mean_generator_loss = 0
        step += 1
