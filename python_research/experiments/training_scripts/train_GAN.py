import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from python_research.augmentation.GAN.discriminator import Discriminator
from python_research.augmentation.GAN.generator import Generator
from python_research.augmentation.GAN.classifier import Classifier
from python_research.augmentation.GAN.WGAN import WGAN
from python_research.dataset_structures import HyperspectralDataset
from python_research.dataset_structures import OrderedDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, help='Path to the dataset in .npy format')
parser.add_argument('--gt_path', type=str, help='Path to the ground truth file in .npy format')
parser.add_argument('--artifacts_path', type=str, help='Path in which artifacts will be stored')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_critic', type=int, default=4, help='number of training steps for discriminator per iter')
parser.add_argument('--patience', type=int, default=200, help='Number of epochs without improvement on generator loss after which training will be terminated')
parser.add_argument('--classifier_patience', type=int, default=15, help='Number of epochs without improvement on classifier loss after which training will be terminated')
parser.add_argument('--verbose', type=bool, help="If True, metric will be printed after each epoch")
parser.add_argument('--classes_count', type=int, default=0, help='Number of classes present in the dataset, if 0 then this count is deduced from the data')
parser.add_argument('--generator_checkout', type=int, default=100, help='Number of epochs after which the PCa plot for generator will be saved')
parser.add_argument('--learning_rate', type=float, default=0.00001, help="Learning rate used for optimizing the parameters")
parser.add_argument('--lambda_gp', type=int, default=10)
parser.add_argument('--b1', type=float, default=0)
parser.add_argument('--b2', type=float, default=0.9)
args = parser.parse_args()
if args.verbose:
    print(args)

os.makedirs(args.artifacts_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

transformed_dataset = HyperspectralDataset(args.dataset_path, args.gt_path)
transformed_dataset.normalize_min_max()
dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

input_shape = bands_count = transformed_dataset.get_data().shape[-1]
if args.classes_count == 0:
    classes_count = len(np.unique(transformed_dataset.get_labels()))
else:
    classes_count = args.classes_count


classifier_criterion = nn.CrossEntropyLoss()
# Initialize generator, discriminator and classifier
generator = Generator(input_shape, classes_count)
discriminator = Discriminator(input_shape)
classifier = Classifier(classifier_criterion, input_shape, classes_count,
                        use_cuda=cuda, patience=args.classifier_patience)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
optimizer_C = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    classifier = classifier.cuda()
    classifier_criterion = classifier_criterion.cuda()

classifier.train_(dataloader, optimizer_C, args.n_epochs)

dataloader = OrderedDataLoader(transformed_dataset, args.batch_size)

gan = WGAN(generator, discriminator, classifier, optimizer_G, optimizer_D,
           use_cuda=cuda, lambda_gp=args.lambda_gp, critic_iters=args.n_critic,
           patience=args.patience, summary_writer=SummaryWriter(args.artifacts_path),
           generator_checkout=args.generator_checkout)
gan.train(dataloader, args.n_epochs, bands_count, args.batch_size,
          classes_count, args.artifacts_path)
