import argparse
import os
import numpy as np
from torch.autograd import Variable

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.autograd as autograd
import torch

from python_research.augmentation.discriminator import Discriminator
from python_research.augmentation.generator import Generator
from python_research.augmentation.classifier import Classifier
from python_research.augmentation.dataset import HyperspectralDataset

os.makedirs('GAN', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='Path to the dataset in .npy format')
parser.add_argument('--gt_path', type=str, help='Path to the ground truth file in .npy format')
parser.add_argument('--artifacts_path', type=str, help='Path in which artifacts will be stored')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=2, help='number of training steps for discriminator per iter')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

CLASS_LABEL = 1

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10

# Configure data loader
transformed_dataset = HyperspectralDataset(opt.dataset_path, opt.gt_path)
dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

input_shape = transformed_dataset.x.shape[-1]

# Initialize generator and discriminator
generator = Generator(input_shape + CLASS_LABEL)
discriminator = Discriminator(input_shape)
classifier = Classifier(input_shape, len(transformed_dataset.classes))

if cuda:
    generator.cuda()
    discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.001, centered=True)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.001, centered=True)
optimizer_C = torch.optim.RMSprop(classifier.parameters(), lr=0.001, centered=True)
classifier_criterion = nn.CrossEntropyLoss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Training
# ----------


d_losses = []
c_losses = []
g_losses = []
batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (samples, labels) in enumerate(dataloader):

        # Configure input
        real_samples = Variable(samples.type(torch.FloatTensor))
        batch_size = len(real_samples)
        labels = Variable(labels.type(torch.LongTensor))
        # ---------------------
        #  Train Discriminator and Classifier
        # ---------------------

        optimizer_D.zero_grad()
        optimizer_C.zero_grad()

        # Sample noise as generator input
        noise = Variable(Tensor(np.random.normal(0, 1, (batch_size, input_shape))).type(torch.FloatTensor))

        # Generate a batch of samples
        reshaped_labels = torch.reshape(labels, (labels.shape[0], 1)).type(torch.FloatTensor)
        noise_with_labels = torch.cat([noise, reshaped_labels], dim=1)
        fake_samples = generator(noise_with_labels)

        # Real samples
        real_discriminator_validity = discriminator(real_samples)
        real_classifier_validity = classifier(Variable(real_samples.data, requires_grad=True))

        # Fake samples
        fake_discriminator_validity = discriminator(fake_samples)
        fake_classifier_validity = classifier(Variable(fake_samples.data, requires_grad=True))
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data)
        # Adversarial loss
        c_loss = classifier_criterion(real_classifier_validity, labels) + classifier_criterion(fake_classifier_validity, labels)
        d_loss = -torch.mean(real_discriminator_validity) + torch.mean(fake_discriminator_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        c_loss.backward()

        optimizer_D.step()
        optimizer_C.step()
        d_losses.append(d_loss.item())
        c_losses.append(c_loss.item())
        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            optimizer_G.zero_grad()
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_samples = generator(noise_with_labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_discriminator_validity = discriminator(fake_samples)
            g_loss = -torch.mean(fake_discriminator_validity)

            g_loss.backward()
            optimizer_G.step()
            g_losses.append(g_loss.item())

            batches_done += opt.n_critic
    print("[Epoch {}/{}] [D loss {}] [G loss {}] [C loss {}]".format(epoch, opt.n_epochs, np.average(d_losses), np.average(g_losses), np.average(c_losses)))

os.makedirs(opt.artifacts_path, exist_ok=True)
torch.save(generator.state_dict(), os.path.join(opt.artifacts_path, "generator_model"))
