import os.path
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from testing_generator import plot_distribution

class WGAN:
    def __init__(self, generator: nn.Module,
                 discriminator: nn.Module,
                 classifier: nn.Module,
                 generator_optimizer: Optimizer,
                 discriminator_optimizer: Optimizer,
                 use_cuda: bool=False,
                 lambda_gp: int=10,
                 critic_iters: int=5,
                 patience: int=None,
                 summary_writer: SummaryWriter=None,
                 verbose: bool=True,
                 generator_checkout: int=None):

        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.use_cuda = use_cuda
        self.losses = {'G': [], 'D': [], 'Real': [], 'Fake': [], 'GP': [], 'GC': []}
        self.lambda_gp = lambda_gp
        self.critic_iters = critic_iters
        self.verbose = verbose
        self.patience = patience
        self.summary_writer = summary_writer
        self.epochs_without_improvement = 0
        self.best_discriminator_loss = np.inf
        self.generator_checkout = generator_checkout

    def _gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1)))
        if self.use_cuda:
            alpha = alpha.cuda()
        # Get random interpolation between real and fake samples
        interpolates = Variable(alpha * real_samples + ((1 - alpha) * fake_samples), requires_grad=True)
        if self.use_cuda:
            interpolates = interpolates.cuda()
        d_interpolates = self.discriminator(interpolates)
        fake = Variable(real_samples.new_full((real_samples.size()[0], 1), 1.0), requires_grad=False)
        if self.use_cuda:
            fake = fake.cuda()
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _discriminator_iteration(self, real_samples: Variable, labels: Variable, noise: Variable):
        with torch.no_grad():
            fake_samples = self.generator(noise, labels)

        real_validity = self.discriminator(real_samples)
        fake_validity = self.discriminator(fake_samples)

        gradient_penalty = self._gradient_penalty(real_samples, fake_samples)
        self.discriminator_optimizer.zero_grad()
        loss = fake_validity.mean() - real_validity.mean() + gradient_penalty
        loss.backward()
        self.discriminator_optimizer.step()

        if self.verbose:
            self.losses['D'].append(loss.item())
            self.losses['Real'].append(real_validity.mean().item())
            self.losses['Fake'].append(fake_validity.mean().item())
            self.losses['GP'].append(gradient_penalty.item())

    def _generator_iteration(self, noise, labels, labels_one_hot):
        self.generator_optimizer.zero_grad()

        fake_samples = self.generator(noise, labels_one_hot)

        fake_discriminator_validity = self.discriminator(fake_samples)
        fake_discriminator_validity = -fake_discriminator_validity.mean()

        fake_classifier_validity = self.classifier(fake_samples)
        fake_classifier_validity = self.classifier.criterion(fake_classifier_validity, labels)
        fake_classifier_validity = fake_classifier_validity.mean()

        loss = fake_discriminator_validity + fake_classifier_validity
        loss.backward()

        self.generator_optimizer.step()
        if self.verbose:
            self.losses['G'].append(loss.item())
            self.losses['GC'].append(fake_classifier_validity.item())

    def _train_epoch(self, data_loader: DataLoader,
                     bands_count: int,
                     batch_size: int,
                     classes_count: int):
        labels_one_hot = torch.zeros([batch_size, classes_count]).type(torch.FloatTensor)
        if self.use_cuda:
            labels_one_hot = labels_one_hot.cuda()

        for parameter in self.generator.parameters():
            parameter.requires_grad = False

        for i, (samples, labels) in enumerate(data_loader):
            real_samples = Variable(samples).type(torch.FloatTensor)
            batch_size = len(real_samples)
            labels = Variable(labels.view(-1, 1).type(torch.LongTensor))

            noise = self._generate_noise(batch_size, bands_count)
            if self.use_cuda:
                real_samples = real_samples.cuda()
                noise = noise.cuda()
                labels = labels.cuda()

            labels_one_hot.scatter_(1, labels, 1)
            self._discriminator_iteration(real_samples, labels_one_hot, noise)

            if i % self.critic_iters == 0:

                for parameter in self.generator.parameters():
                    parameter.requires_grad = True

                noise = self._generate_noise(batch_size, bands_count)
                if self.use_cuda:
                    noise = noise.cuda()
                    labels = labels.cuda()

                self._generator_iteration(noise, labels.view(labels.shape[0]), labels_one_hot)

                for parameter in self.generator.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _generate_noise(batch_size, bands_count):
        noise = torch.FloatTensor(np.random.normal(0.5, 0.1, (batch_size, bands_count)))
        return Variable(noise)

    def _print_metrics(self, epoch: int):
        generator_loss = np.average(self.losses['G'])
        discriminator_loss = np.average(self.losses['D'])
        real = np.average(self.losses['Real'])
        fake = np.average(self.losses['Fake'])
        gc = np.average(self.losses['GC'])
        gp = np.average(self.losses['GP'])
        self.summary_writer.add_scalars('GAN', {'D': discriminator_loss,
                                                'G': generator_loss}, epoch)
        print("[Epoch: {}][D loss: {}] [G loss: {}] "
              "[R: {}] [F: {}] [GP: {}] [GC: {}]".format(epoch, discriminator_loss,
                                                         generator_loss, real, fake, gp, gc))

    def _save_generator(self, path, epoch=None, name=None):
        if name is not None:
            final_path = os.path.join(path, name)
        elif epoch is not None:
            final_path = os.path.join(path, 'generator_model_epoch_{}'.format(epoch))
        else:
            final_path = os.path.join(path, 'generator_model')
        torch.save(self.generator.state_dict(), final_path)

    def _zero_losses(self):
        for loss in self.losses:
            self.losses[loss].clear()

    def _early_stopping(self) -> bool:
        if abs(np.average(self.losses['D'])) < self.best_discriminator_loss:
            self.best_discriminator_loss = abs(np.average(self.losses['D']))
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                if self.verbose:
                    print("{} epochs without improvement, terminating".format(self.patience))
                return True
            return False

    def train(self, dataset,
              data_loader: DataLoader,
              epochs: int,
              bands_count: int,
              batch_size: int,
              classes_count: int,
              artifacts_path: str):

        for parameter in self.classifier.parameters():
            parameter.requires_grad = False

        for epoch in range(epochs):
            self._train_epoch(data_loader, bands_count, batch_size, classes_count)
            if self.patience is not None:
                if self._early_stopping():
                    break
            if self.verbose:
                self._print_metrics(epoch)
            if self.generator_checkout:
                if epoch % self.generator_checkout == 0:
                    figure_path = os.path.join(artifacts_path, 'pca_plot_epoch_{}'.format(epoch))
                    device = 'gpu' if self.use_cuda else 'cpu'
                    self._save_generator(artifacts_path, epoch)
                    plot_distribution(dataset, os.path.join(artifacts_path, 'generator_model_epoch_{}'.format(epoch)),
                                      figure_path, bands_count, classes_count, device=device)
            self._zero_losses()
