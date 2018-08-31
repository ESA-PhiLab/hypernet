import argparse
import os
import numpy as np
from time import time

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.autograd as autograd
import torch
from tensorboardX import SummaryWriter

from python_research.augmentation.discriminator import Discriminator
from python_research.augmentation.generator import Generator
from python_research.augmentation.classifier import Classifier
from python_research.augmentation.dataset import HyperspectralDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='Path to the dataset in .npy format')
parser.add_argument('--gt_path', type=str, help='Path to the ground truth file in .npy format')
parser.add_argument('--artifacts_path', type=str, help='Path in which artifacts will be stored')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument('--patience', type=int, default=200, help='Number of epochs without improvement on generator loss after which training will be terminated')
parser.add_argument('--classifier_patience', type=int, default=15, help='Number of epochs without improvement on classifier loss after which training will be terminated')
parser.add_argument('--verbose', type=bool, help="If True, metric will be printed after each epoch")
parser.add_argument('--classes_count', type=int, default=0, help='Number of classes present in the dataset, if 0 then this count is deduced from the data')
args = parser.parse_args()
if args.verbose:
    print(args)

METRICS_PATH = os.path.join(args.artifacts_path, "metrics.csv")
BEST_MODEL_PATH = os.path.join(args.artifacts_path, "best_model")
LAST_MODEl_PATH = os.path.join(args.artifacts_path, "last_model")
writer = SummaryWriter(args.artifacts_path)

os.makedirs(args.artifacts_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
# Loss weight for gradient penalty
lambda_gp = 10

# Configure data loader
transformed_dataset = HyperspectralDataset(args.dataset_path, args.gt_path, samples_per_class=0)
dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

input_shape = transformed_dataset.x.shape[-1]
if args.classes_count == 0:
    classes_count = len(np.unique(transformed_dataset.y))
else:
    classes_count = args.classes_count

# Initialize generator, discriminator and classifier
generator = Generator(input_shape, classes_count)
discriminator = Discriminator(input_shape)
classifier = Classifier(input_shape, classes_count)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.01)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
optimizer_C = torch.optim.Adam(classifier.parameters(), lr=0.001)
classifier_criterion = nn.CrossEntropyLoss()

if cuda:
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    classifier_criterion.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
max_label = float(max(np.unique(transformed_dataset.y)))
max_label = FloatTensor([max_label])


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

# Train classifier first
best_classifier_loss = np.inf
epochs_without_improvement = 0
for epoch in range(args.n_epochs):
    c_losses = []
    for i, (samples, labels) in enumerate(dataloader):
        real_samples = Variable(samples.type(FloatTensor))
        batch_size = len(real_samples)
        labels = Variable(labels.type(LongTensor))
        optimizer_C.zero_grad()
        real_classifier_validity = classifier(Variable(real_samples.data, requires_grad=True))
        # real_classifier_validity = real_classifier_validity.view(64)
        c_loss = classifier_criterion(real_classifier_validity, labels)
        c_loss.backward()
        optimizer_C.step()
        c_losses.append(c_loss.item())
    classifier_loss = np.average(c_losses)
    if args.verbose:
        print("[C loss: {}]".format(classifier_loss))
    if classifier_loss < best_classifier_loss:
        epochs_without_improvement = 0
        best_classifier_loss = classifier_loss
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= args.classifier_patience:
            print("{} epochs without improvement for classifier, terminating".format(args.classifier_patience))
            break

one = FloatTensor([1])
mone = one * -1
# Train discriminator and generator
best_loss = np.inf
epochs_without_improvement = 0
labels_one_hot = FloatTensor(args.batch_size, classes_count)

for p in classifier.parameters():
    p.requires_grad = False

for epoch in range(args.n_epochs):
    d_losses = []
    g_losses = []
    real = []
    fake = []
    g_fake = []
    g_class = []
    epoch_start = time()
    for i, (samples, labels) in enumerate(dataloader):
        # Configure input
        real_samples = Variable(samples.type(FloatTensor))
        batch_size = len(real_samples)
        labels = Variable(labels.type(LongTensor))
        # ---------------------
        #  Train Discriminator
        # ---------------------

        for p in discriminator.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        # Sample noise as generator input
        noise = FloatTensor(np.random.normal(0.5, 0.1, (batch_size, input_shape)))

        # Generate a batch of samples
        reshaped_labels = labels.view(batch_size, 1)
        labels_one_hot.scatter_(1, reshaped_labels, 1)
        noise_with_labels = Variable(torch.cat([noise, labels_one_hot], dim=1))
        with torch.no_grad():
            fake_samples = Variable(generator(noise_with_labels).data)

        # Real samples
        real_discriminator_validity = discriminator(real_samples)
        real_discriminator_validity = real_discriminator_validity.mean()
        # real_discriminator_validity.backward()
        # Fake samples
        fake_discriminator_validity = discriminator(fake_samples)
        fake_discriminator_validity = fake_discriminator_validity.mean()
        # fake_discriminator_validity.backward()

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data) * lambda_gp
        # gradient_penalty.backward()
        # Adversarial loss
        # d_loss = -torch.mean(real_discriminator_validity) + torch.mean(fake_discriminator_validity)# + lambda_gp * gradient_penalty
        d_loss = fake_discriminator_validity - real_discriminator_validity + gradient_penalty
        d_loss.backward()
        optimizer_D.step()
        real.append(torch.mean(real_discriminator_validity).cpu().detach().numpy())
        fake.append(torch.mean(fake_discriminator_validity).cpu().detach().numpy())
        d_losses.append(d_loss.item())

        # for p in discriminator.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        # Train the generator every n_critic steps
        if i % args.n_critic == 0:

            for p in discriminator.parameters():
                p.requires_grad = False


            optimizer_G.zero_grad()
            # -----------------
            #  Train Generator
            # -----------------
            noise = FloatTensor(np.random.normal(0.5, 0.1, (batch_size, input_shape)))
            noise_with_labels = Variable(torch.cat([noise, labels_one_hot], dim=1))
            # Generate a batch of samples
            fake_samples = generator(noise_with_labels)
            # Loss measures generator's ability to fool the discriminator and generate samples from correct class
            # Train on fake images
            fake_discriminator_validity = discriminator(fake_samples)
            fake_discriminator_validity = -fake_discriminator_validity.mean()
            fake_discriminator_validity.backward()

            fake_samples = generator(noise_with_labels)
            fake_classifier_validity = classifier(fake_samples)
            fake_classifier_validity = classifier_criterion(fake_classifier_validity, labels)
            fake_classifier_validity.backward()
            optimizer_G.step()
            g_loss = fake_discriminator_validity.item() + fake_classifier_validity.item()
            # g_loss.backward()
            g_fake.append(fake_discriminator_validity.item())
            g_class.append(fake_classifier_validity.item())
            g_losses.append(g_loss)

    epoch_duration = time() - epoch_start
    metrics = open(METRICS_PATH, 'a')
    metrics.write("{},{},{},{}\n".format(epoch,
                                         epoch_duration,
                                         np.average(d_losses),
                                         np.average(g_losses)))
    metrics.close()
    generator_loss = abs(np.average(d_losses))
    if best_loss > generator_loss:
        torch.save(generator.state_dict(), BEST_MODEL_PATH)
        best_loss = generator_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print("{} epochs without improvement, terminating".format(args.patience))
            break
    writer.add_scalars('GAN', {'D': np.average(d_losses),
                               'G': np.average(g_losses)}, epoch)
    torch.save(generator.state_dict(), LAST_MODEl_PATH)
    if args.verbose:
        print("[Epoch {}/{}] [D loss {}] [G loss {}] [real: {} fake: {}] [g_fake {} g_class {}]".format(epoch,
                                                                         args.n_epochs,
                                                                         np.average(d_losses),
                                                                         np.average(g_losses),
                                                                         np.average(real),
                                                                         np.average(fake),
                                                                         np.average(g_fake),
                                                                         np.average(g_class)))


