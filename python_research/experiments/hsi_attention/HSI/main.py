import argparse
import os
import pickle
import shutil
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(r'C:\Users\ltulczyjew\Desktop\software')

from python_research.experiments.hsi_attention.HSI.datasets.generate_trained_models import get_loader_function, \
    produce_splits
from python_research.experiments.hsi_attention.HSI.models.model_2 import Model2
from python_research.experiments.hsi_attention.HSI.models.model_3 import Model3
from python_research.experiments.hsi_attention.HSI.models.model_4 import Model4


def arguments():
    parser = argparse.ArgumentParser(description='Input  arguments.')

    parser.add_argument('--dataset',
                        action="store",
                        dest="dataset",
                        type=str,
                        help='Dataset')

    parser.add_argument('--selected_bands',
                        dest='selected_bands',
                        type=str)

    parser.add_argument('--validation',
                        action="store",
                        dest="validation_proportion",
                        type=float,
                        help='Proportion of validation samples', default=0.1)

    parser.add_argument('--test',
                        action="store",
                        dest="test_proportion",
                        type=float,
                        help='Proportion of test samples', default=0.1)

    parser.add_argument('--epochs',
                        action="store",
                        dest="epochs",
                        type=int,
                        help='Number of epochs',
                        default=999999)

    parser.add_argument('--modules',
                        action="store",
                        dest="modules",
                        type=int,
                        help='Number of attention modules',
                        default=2)

    parser.add_argument('--patience',
                        action="store",
                        dest="patience",
                        type=int,
                        help='Patience',
                        default=6)

    parser.add_argument('--output',
                        action="store",
                        dest="output_dir",
                        type=str,
                        help='Output directory')

    parser.add_argument('--batch_size',
                        action="store",
                        dest="batch_size",
                        type=int,
                        help='batch size',
                        default=200)

    parser.add_argument('--uses_attention',
                        action="store",
                        dest="uses_attention",
                        type=bool,
                        help='batch size',
                        default=True)

    return parser.parse_args()


def build_input_tensors(x, y):
    x = np.asarray(x, dtype=np.float).reshape(1, 1, x.size)
    y = np.asarray(y, dtype=np.float).reshape(1, y.size)
    x = torch.from_numpy(x)
    y = torch.from_numpy(np.asarray(y))
    x = x.type(torch.FloatTensor)
    y = y.type(torch.FloatTensor)
    x = torch.autograd.Variable(x)

    return x, y


def train_network(x_train, y_train, x_val, y_val, model, epochs, args):
    training_history = []
    validation_history = []
    loss_history = []
    training_time = []
    validation_time = []

    for current_epoch in range(epochs):
        for param in model.parameters():
            param.requires_grad = True

        print("Epoch {}:".format(current_epoch))

        print("\tTraining:")

        training_accuracies = []
        losses = []

        begin = time.time()

        x_train_list = [x_train[i:i + args.batch_size] for i in range(0, len(x_train), args.batch_size)]
        y_train_list = [y_train[i:i + args.batch_size] for i in range(0, len(y_train), args.batch_size)]

        for x, y in tqdm(zip(x_train_list, y_train_list), total=len(x_train_list)):
            x = torch.from_numpy(x.astype('int32')).type(torch.FloatTensor)
            y = torch.from_numpy(y.astype('int32')).type(torch.FloatTensor)

            model.zero_grad()
            model._optimizer.zero_grad()

            out = model(x)

            loss = model._loss(out, y)
            losses.append(loss.data)

            accuracy = (torch.argmax(out, dim=1) == torch.argmax(y.reshape(-1, y.shape[2]), dim=1)).sum().type(
                torch.DoubleTensor) / y.shape[0]

            training_accuracies.append(accuracy)

            loss.backward()
            model._optimizer.step()

        training_time.append(time.time() - begin)

        validation_accuracies = []

        print("\tValidation:")

        begin = time.time()

        for param in model.parameters():
            param.requires_grad = False

        x_val_list = [x_val[i:i + args.batch_size] for i in range(0, len(x_val), args.batch_size)]
        y_val_list = [y_val[i:i + args.batch_size] for i in range(0, len(y_val), args.batch_size)]

        for x, y in tqdm(zip(x_val_list, y_val_list), total=len(x_val_list)):
            x = torch.from_numpy(x.astype('int32')).type(torch.FloatTensor)
            y = torch.from_numpy(y.astype('int32')).type(torch.FloatTensor).reshape(-1, y.shape[2])

            accuracy = (torch.argmax(model(x), dim=1) == torch.argmax(y, dim=1)).sum().type(torch.DoubleTensor) / \
                       y.shape[0]

            validation_accuracies.append(accuracy)

        validation_time.append(time.time() - begin)

        loss_history.append(np.mean(losses))
        training_history.append(np.mean(training_accuracies))
        validation_history.append(np.mean(validation_accuracies))

        print('\tLoss: {0}% '.format(loss_history[-1]))
        print('\tTraining accuracy: {0}% '.format(training_history[-1]))
        print('\tValidation accuracy: {0}% '.format(validation_history[-1]))

        pickle.dump(training_history,
                    open(args.output_dir + '\\' + args.dataset + '_training_history.pkl', 'wb'))
        pickle.dump(validation_history,
                    open(args.output_dir + '\\' + args.dataset + '_validaton_history.pkl', 'wb'))
        pickle.dump(loss_history, open(args.output_dir + '\\' + args.dataset + '_loss_history.pkl', 'wb'))
        pickle.dump(training_time,
                    open(args.output_dir + '\\' + args.dataset + '_time_training.pkl', 'wb'))
        pickle.dump(validation_time,
                    open(args.output_dir + '\\' + args.dataset + '_time_validation.pkl', 'wb'))

        if current_epoch == 0 or (validation_history[-1] > max(validation_history[:-1])):
            print('\tSaving model...\n')
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            torch.save(model, args.output_dir + '\\' + args.dataset + '_model.pt')

        # Should we finish prematurely?
        if current_epoch > args.patience:
            if max(validation_history[:-args.patience]) > max(validation_history[-args.patience:]):
                # bail
                break


def test_network(x_test, y_test, args):
    model = torch.load(args.output_dir + '\\' + args.dataset + '_model.pt')

    testing_accuracies = []

    for param in model.parameters():
        param.requires_grad = False

    print('\tTesting:')

    begin = time.time()

    x_test_list = [x_test[i:i + args.batch_size] for i in range(0, len(x_test), args.batch_size)]
    y_test_list = [y_test[i:i + args.batch_size] for i in range(0, len(y_test), args.batch_size)]

    heatmaps_per_class = [[] for _ in range(y_test.shape[-1])]

    for x, y in tqdm(zip(x_test_list, y_test_list), total=len(x_test_list)):
        x = torch.from_numpy(x.astype('int32')).type(torch.FloatTensor)
        y = torch.from_numpy(y.astype('int32')).type(torch.FloatTensor).reshape(-1, y.shape[2])

        accuracy = (torch.argmax(model(x), dim=1) == torch.argmax(y, dim=1)).sum().type(torch.DoubleTensor) / y.shape[0]

        heatmaps = model.get_heatmaps(x.shape[-1])

        for i in range(y.shape[0]):
            heatmaps_per_class[torch.argmax(y[i, :])].extend([h[i] for h in heatmaps])

        testing_accuracies.append(accuracy)

    for i in range(len(heatmaps_per_class)):
        heatmaps_per_class[i] = np.mean(np.array(heatmaps_per_class[i]), axis=0)[0]

    testing_time = time.time() - begin
    testing_accuracy = [np.mean(testing_accuracies)]

    print('\tTesting accuracy: {0}% '.format(testing_accuracy))

    pickle.dump(testing_accuracy,
                open(args.output_dir + '\\' + args.dataset + '_testing_accuracy.pkl', 'wb'))
    pickle.dump(testing_time, open(args.output_dir + '\\' + args.dataset + '_time_testing.pkl', 'wb'))
    pickle.dump(heatmaps_per_class,
                open(args.output_dir + '\\' + args.dataset + '_attention_bands.pkl', 'wb'))


def load_model(n_attention_modules, n_classes, input_dimension):
    if n_attention_modules == 2:
        return Model2(n_classes, input_dimension)
    if n_attention_modules == 3:
        return Model3(n_classes, input_dimension)
    if n_attention_modules == 4:
        return Model4(n_classes, input_dimension)

    raise Exception


def run(args):
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)

    print("Training attention model for dataset: {}".format(args.dataset))

    samples, labels = get_loader_function(args.dataset)()

    # # Select bands:
    # selected_bands_ids = np.loadtxt(args.selected_bands).astype(int)
    # samples = samples[selected_bands_ids, ...]
    print('Spectral size: {}'.format(samples.shape[-1]))
    (x_train, y_train), \
    (x_val, y_val), \
    (x_test, y_test) = produce_splits(samples,
                                      labels,
                                      args.validation_proportion,
                                      args.test_proportion)

    model = load_model(args.modules, y_train.shape[-1], x_train.shape[-1])
    model.to(device)

    model.with_attention(args.uses_attention)

    train_network(x_train,
                  y_train,
                  x_val,
                  y_val,
                  model,
                  args.epochs, args)

    test_network(x_test, y_test, args)


if __name__ == "__main__":
    args = arguments()
    run(args)
