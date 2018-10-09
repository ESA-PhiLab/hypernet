import os
import argparse
from copy import copy

import torch
import numpy as np
import torch.nn as nn
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from python_research.experiments.multiple_feature_learning.builders.keras_builders import \
    build_1d_model
from python_research.experiments.multiple_feature_learning.utils.utils import load_patches
from python_research.experiments.multiple_feature_learning.utils.dataset import Dataset
from python_research.experiments.multiple_feature_learning.utils.keras_custom_callbacks import \
    TimeHistory
from python_research.augmentation.dataset import HyperspectralDataset, CustomDataLoader
from python_research.augmentation.classifier import Classifier
from python_research.augmentation.discriminator import Discriminator
from python_research.augmentation.generator import Generator
from python_research.augmentation.WGAN import WGAN
from python_research.augmentation.augmentation import generate_samples


def parse_args():
    parser = argparse.ArgumentParser()
    # Learning parameters
    parser.add_argument('--dataset_file', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--gt_file', type=str,
                        help="Path to the ground truth in .npy format")
    parser.add_argument("--output_dir", type=str, default="train_grids_output",
                        help="Path to the output directory in which artifacts will be stored")
    parser.add_argument("--output_file", type=str, default="run01",
                        help="Name of the output file in which data will be stored")
    parser.add_argument("--runs", type=int, default=10,
                        help="How many times to run the validation")
    parser.add_argument("--training_samples_per_class", type=int, default=250,
                        help="Number of train samples per class to use")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of training batch')
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs without improvement on validation score before '
                             'stopping the learning')
    parser.add_argument('--kernels', type=int, default=200,
                        help='Number of kernels in fir convolutional layer')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Number of epochs without improvement on validation score before '
                             'stopping the learning')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity of training')
    # WGAN parameters
    parser.add_argument('--n_epochs_gan', type=int, default=200,
                        help='number of epochs of training for GAN')
    parser.add_argument('--n_critic', type=int, default=4,
                        help='number of training steps for discriminator per iter')
    parser.add_argument('--patience_gan', type=int, default=200,
                        help='Number of epochs without improvement on discriminator loss after which training will be terminated')
    parser.add_argument('--classifier_patience', type=int, default=15,
                        help='Number of epochs without improvement on classifier loss after which training will be terminated')
    parser.add_argument('--classes_count', type=int, default=0,
                        help='Number of classes present in the dataset, if 0 then this count is deduced from the data')
    parser.add_argument('--generator_checkout', type=int, default=100,
                        help='Number of epochs after which the PCa plot for generator will be saved')
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
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Load patches
    data, test = load_patches(args.dataset_file, args.classes_count, [1,1])

    # Normalize data
    data.normalize_data(args.classes_count)
    test.x_test = (test.x_test.astype(np.float64) - data.min) / (data.max - data.min)
    data_to_augment_on = HyperspectralDataset(data.x_train[:, :, 0], np.argmax(data.y_train, axis=1),
                                              normalize=False)
    custom_data_loader = CustomDataLoader(data_to_augment_on, args.batch_size)
    data_loader = DataLoader(data_to_augment_on, batch_size=args.batch_size, shuffle=True,
                             drop_last=True)

    cuda = True if torch.cuda.is_available() else False

    input_shape = bands_count = data_to_augment_on.x.shape[-1]
    if args.classes_count == 0:
        classes_count = len(np.unique(data_to_augment_on.y))
    else:
        classes_count = args.classes_count

    classifier_criterion = nn.CrossEntropyLoss()
    # Initialize generator, discriminator and classifier
    generator = Generator(input_shape, classes_count)
    discriminator = Discriminator(input_shape)
    classifier = Classifier(classifier_criterion, input_shape, classes_count,
                            use_cuda=cuda, patience=args.classifier_patience)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate,
                                   betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate,
                                   betas=(args.b1, args.b2))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                   betas=(args.b1, args.b2))

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        classifier = classifier.cuda()
        classifier_criterion = classifier_criterion.cuda()

    classifier.train_(data_loader, optimizer_C, args.n_epochs_gan)

    gan = WGAN(generator, discriminator, classifier, optimizer_G, optimizer_D,
               use_cuda=cuda, lambda_gp=args.lambda_gp, critic_iters=args.n_critic,
               patience=args.patience_gan, summary_writer=SummaryWriter(args.output_dir),
               generator_checkout=args.generator_checkout)
    gan.train(copy(data_to_augment_on), custom_data_loader, args.n_epochs_gan, bands_count,
              args.batch_size, classes_count, os.path.join(args.output_dir, args.output_file) + "generator_model")

    samples_per_class = get_samples_per_class_count(data_to_augment_on.y)
    generated_x, generated_y = generate_samples(gan.generator, samples_per_class, bands_count, classes_count)

    generated_x = np.reshape(generated_x.numpy(), generated_x.shape + (1, ))
    generated_y = to_categorical(generated_y, classes_count)

    data.x_train = np.concatenate([data.x_train, generated_x], axis=0)
    data.y_train = np.concatenate([data.y_train, generated_y], axis=0)

    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_file) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Build model
    model = build_1d_model((data.x_train.shape[1], 1), args.kernels, args.kernel_size,
                           args.classes_count)

    # Train model
    history = model.fit(x=data.x_train,
                        y=data.y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=[early, logger, checkpoint, timer],
                        validation_data=(data.x_val, data.y_val))

    # Load best model
    model = load_model(os.path.join(args.output_dir, args.output_file) + "_model")

    # Calculate test set score
    test_score = model.evaluate(x=test.x_test,
                                y=test.y_test)

    # Calculate accuracy for each class
    predictions = model.predict(x=test.x_test)
    predictions = np.argmax(predictions, axis=1)
    y_true = np.argmax(test.y_test, axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=np.unique(y_true))
    matrix = matrix / matrix.astype(np.float).sum(axis=1)
    class_accuracy = np.diagonal(matrix)
    # Collect metrics
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    avg_epoch_time = np.average(np.array(timer.average))
    epochs = len(history.epoch)

    # Save metrics
    metrics = open(os.path.join(args.output_dir, "metrics.csv"), 'a')
    class_accuracy_csv = open(os.path.join(args.output_dir, "class_accuracy.csv"), 'a')
    metrics.write(
        str(train_score) + "," + str(val_score) + "," + str(test_score[1]) + "," + str(
            time) + "," + str(epochs) + "," + str(avg_epoch_time) + "\n")
    class_accuracy_csv.write(",".join(str(x) for x in class_accuracy) + "\n")
    metrics.close()
    class_accuracy_csv.close()
    np.savetxt(os.path.join(args.output_dir, args.output_file) + "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parse_args()
    for j in range(0, 5):
        args.dataset_file = args.dataset_file[:-1] + str(j)
        args.output_dir = args.output_dir[:-1] + str(j)
        for i in range(1, 6):
            args.output_file = args.output_file[:-1] + str(i)
            main(args)