import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from keras.models import load_model
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from utils import calculate_class_accuracy
from python_research.io import save_to_csv
from python_research.keras_models import \
    build_1d_model
from python_research.keras_custom_callbacks import \
    TimeHistory
from python_research.dataset_structures import BalancedSubset, \
    ImbalancedSubset, CustomSizeSubset
from python_research.dataset_structures import HyperspectralDataset
from python_research.dataset_structures import OrderedDataLoader
from python_research.augmentation.GAN.classifier import Classifier
from python_research.augmentation.GAN.discriminator import Discriminator
from python_research.augmentation.GAN.generator import Generator
from python_research.augmentation.GAN.WGAN import WGAN
from python_research.augmentation.GAN.samples_generator import SamplesGenerator

def parse_args():
    parser = argparse.ArgumentParser()
    # Learning parameters
    parser.add_argument('--dataset_path', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--gt_path', type=str,
                        help="Path to the ground truth in .npy format")
    parser.add_argument("--artifacts_path", type=str, default="artifacts",
                        help="Path to the output directory in which "
                             "artifacts will be stored")
    parser.add_argument("--output_file", type=str, default="run01",
                        help="Name of the output file in which data "
                             "will be stored")
    parser.add_argument("--classes_count", type=int, default=0,
                        help='Number of classes present in the dataset, if 0 '
                             'then this count is deduced from the data')
    parser.add_argument("--runs", type=int, default=10,
                        help="How many times to run the validation")
    parser.add_argument("--balanced", type=int, default=1,
                        help="Whether each class should have an equal "
                             "number of samples. If True, parameter "
                             "train_samples should be equal to a number of "
                             "samples for each class, if False, paramter "
                             "train_samples should be equal to total number "
                             "of samples in the extracted dataset")
    parser.add_argument("--pixel_neighborhood", type=int, default=1,
                        help="neighborhood of an extracted pixel when "
                             "preparing the data for training and "
                             "classification. This value should define height "
                             "and width simultaneously.  If equals 1, "
                             "only spectral information will be included "
                             "in a sample")
    parser.add_argument("--train_samples", type=int, default=250,
                        help="Number of train samples per class to use")
    parser.add_argument("--val_set_part", type=float, default=0.1,
                        help="Percentage of a training set to be extracted "
                             "as a validation set")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of training batch')
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs without improvement on '
                             'validation score before '
                             'stopping the learning')
    parser.add_argument('--kernels', type=int, default=200,
                        help='Number of kernels in first convolutional layer')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Size of a kernel in first convolution layer')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity of training')
    # WGAN parameters
    parser.add_argument('--n_epochs_gan', type=int, default=1,
                        help='number of epochs of training for GAN')
    parser.add_argument('--n_critic', type=int, default=4,
                        help='number of training steps for discriminator '
                             'per iter')
    parser.add_argument('--patience_gan', type=int, default=200,
                        help='Number of epochs without improvement on '
                             'discriminator loss after which training will '
                             'be terminated')
    parser.add_argument('--classifier_patience', type=int, default=15,
                        help='Number of epochs without improvement on '
                             'classifier loss after which training will be '
                             'terminated')
    parser.add_argument('--generator_checkout', type=int, default=0,
                        help='Number of epochs after which the PCa plot for '
                             'generator will be saved')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help="Learning rate used for optimizing the parameters")
    parser.add_argument('--lambda_gp', type=int, default=10)
    parser.add_argument('--b1', type=float, default=0)
    parser.add_argument('--b2', type=float, default=0.9)
    return parser.parse_args()


def get_samples_per_class_count(y):
    samples_per_class = dict.fromkeys(np.unique(y))
    for label in samples_per_class:
        samples_per_class[label] = len(np.where(y == label)[0])
    return samples_per_class


def main(args):
    os.makedirs(os.path.join(args.artifacts_path), exist_ok=True)
    # Init data
    test_data = HyperspectralDataset(args.dataset_path, args.gt_path,
                                     neighbourhood_size=args.pixel_neighbourhood)
    test_data.normalize_labels()
    if args.balanced == 1:
        train_data = BalancedSubset(test_data, args.train_samples)
        val_data = BalancedSubset(train_data, args.val_set_part)
    elif args.balanced == 0:
        train_data = ImbalancedSubset(test_data, args.train_samples)
        val_data = ImbalancedSubset(train_data, args.val_set_part)
    elif args.balanced == 2:  # Case for balanced indiana
        train_data = CustomSizeSubset(test_data, [30, 250, 250, 150, 250, 250,
                                                  20, 250, 15, 250, 250, 250,
                                                  150, 250, 50, 50])
        val_data = BalancedSubset(train_data, args.val_set_part)
    # Normalize data
    max_ = train_data.max if train_data.max > val_data.max else val_data.max
    min_ = train_data.min if train_data.min < val_data.min else val_data.min
    train_data.normalize_min_max(min_=min_, max_=max_)
    val_data.normalize_min_max(min_=min_, max_=max_)
    test_data.normalize_min_max(min_=min_, max_=max_)

    custom_data_loader = OrderedDataLoader(train_data, args.batch_size)
    data_loader = DataLoader(train_data, batch_size=args.batch_size,
                             shuffle=True, drop_last=True)

    cuda = True if torch.cuda.is_available() else False
    input_shape = bands_count = train_data.shape[-1]
    if args.classes_count == 0:
        args.classes_count = len(np.unique(train_data.get_labels()))

    classifier_criterion = nn.CrossEntropyLoss()
    # Initialize generator, discriminator and classifier
    generator = Generator(input_shape, args.classes_count)
    discriminator = Discriminator(input_shape)
    classifier = Classifier(classifier_criterion, input_shape, args.classes_count,
                            use_cuda=cuda, patience=args.classifier_patience)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.b1, args.b2))
    optimizer_C = torch.optim.Adam(classifier.parameters(),
                                   lr=args.learning_rate,
                                   betas=(args.b1, args.b2))

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        classifier = classifier.cuda()
        classifier_criterion = classifier_criterion.cuda()

    # Train classifier
    classifier.train_(data_loader, optimizer_C, args.n_epochs_gan)

    gan = WGAN(generator, discriminator, classifier, optimizer_G, optimizer_D,
               use_cuda=cuda, lambda_gp=args.lambda_gp, critic_iters=args.n_critic,
               patience=args.patience_gan, summary_writer=SummaryWriter(args.artifacts_path),
               generator_checkout=args.generator_checkout)
    # Train GAN
    gan.train(custom_data_loader, args.n_epochs_gan, bands_count,
              args.batch_size, args.classes_count,
              os.path.join(args.artifacts_path, args.output_file) + "_generator_model")

    # Generate samples using trained Generator
    generator = Generator(input_shape, args.classes_count)
    generator_path = os.path.join(args.artifacts_path, args.output_file + "_generator_model")
    generator.load_state_dict(torch.load(generator_path))
    if cuda:
        generator = generator.cuda()
    train_data.convert_to_numpy()

    device = 'gpu' if cuda is True else 'cpu'
    samples_generator = SamplesGenerator(device=device)
    generated_x, generated_y = samples_generator.generate(train_data,
                                                          generator)
    generated_x = np.reshape(generated_x.detach().cpu().numpy(),
                             generated_x.shape + (1, ))

    train_data.expand_dims(axis=-1)
    test_data.expand_dims(axis=-1)
    val_data.expand_dims(axis=-1)

    train_data.vstack(generated_x)
    train_data.hstack(generated_y)

    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.artifacts_path, args.output_file) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.artifacts_path,
                                              args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Build model
    model = build_1d_model((test_data.shape[1:]), args.kernels,
                            args.kernel_size, args.classes_count)

    # Train model
    history = model.fit(x=train_data.get_data(),
                        y=train_data.get_one_hot_labels(args.classes_count),
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=[early, logger, checkpoint, timer],
                        validation_data=(val_data.get_data(),
                                         val_data.get_one_hot_labels(args.classes_count)))

    # Load best model
    model = load_model(os.path.join(args.artifacts_path, args.output_file) + "_model")

    # Calculate test set score
    test_score = model.evaluate(x=test_data.get_data(),
                                y=test_data.get_one_hot_labels(
                                    args.classes_count))

    # Calculate accuracy for each class
    predictions = model.predict(x=test_data.get_data())
    predictions = np.argmax(predictions, axis=1)
    class_accuracy = calculate_class_accuracy(predictions,
                                              test_data.get_labels(),
                                              args.classes_count)
    # Collect metrics
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    avg_epoch_time = np.average(np.array(timer.average))
    epochs = len(history.epoch)

    # Save metrics
    metrics_path = os.path.join(args.artifacts_path, "metrics.csv")
    save_to_csv(metrics_path, [train_score, val_score,
                               test_score[1], time, epochs, avg_epoch_time])
    class_accuracy_path = os.path.join(args.artifacts_path,
                                       "class_accuracy.csv")
    save_to_csv(class_accuracy_path, class_accuracy)
    np.savetxt(os.path.join(args.artifacts_path, args.output_file) +
               "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parse_args()
    for i in range(0, args.runs):
        if i < 10:
            args.output_file = args.output_file[:-1] + str(i)
        else:
            args.output_file = args.output_file[:-2] + str(i)
        main(args)
