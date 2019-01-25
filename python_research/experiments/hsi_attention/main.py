import argparse
import os
import pickle
import time

import numpy as np
import torch
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm

from python_research.experiments.hsi_attention.arguments import arguments
from python_research.experiments.hsi_attention.datasets.generate_trained_models import get_loader_function, \
    produce_splits
from python_research.experiments.hsi_attention.models.model_2 import Model2
from python_research.experiments.hsi_attention.models.model_3 import Model3
from python_research.experiments.hsi_attention.models.model_4 import Model4
from python_research.experiments.hsi_attention.visualization import plot_heatmaps


def train_network(x_train, y_train, x_val, y_val, model, args):
    """
    Train model.
    :param x_train: Samples for training.
    :param y_train: Labels for training.
    :param x_val: Samples for evaluation.
    :param y_val: Labels for evaluation.
    :param model: Model designed for training.
    :param args: Parsed arguments.
    :return:
    """
    training_history = []
    validation_history = []
    loss_history = []
    training_time = []
    validation_time = []
    for current_epoch in range(int(args.epochs)):
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
            x = torch.from_numpy(x.astype('float32')).unsqueeze(1).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y.astype('int32')).type(torch.cuda.LongTensor)
            model.zero_grad()
            model._optimizer.zero_grad()
            out = model(x)
            loss = model._loss(out, y)
            losses.append(loss.clone().detach().cpu().numpy())
            accuracy = (torch.argmax(out, dim=1) == y).sum().type(torch.cuda.DoubleTensor) / y.shape[0]
            training_accuracies.append(accuracy.cpu().numpy())
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
            x = torch.from_numpy(x.astype('float32')).unsqueeze(1).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y.astype('int32')).type(torch.cuda.LongTensor)
            accuracy = (torch.argmax(model(x), dim=1) == y).sum().type(torch.cuda.DoubleTensor) / y.shape[0]
            validation_accuracies.append(accuracy.cpu().numpy())
        validation_time.append(time.time() - begin)
        loss_history.append(np.mean(losses))
        training_history.append(np.mean(training_accuracies))
        validation_history.append(np.mean(validation_accuracies))
        print('\tLoss: {0}% '.format(loss_history[-1]))
        print('\tTraining accuracy: {0}% '.format(training_history[-1]))
        print('\tValidation accuracy: {0}% '.format(validation_history[-1]))
        pickle.dump(training_history,
                    open(os.path.join(args.output_dir, args.run_idx + '_training_history.pkl'), 'wb'))
        pickle.dump(validation_history,
                    open(os.path.join(args.output_dir, args.run_idx + '_validation_history.pkl'), 'wb'))
        pickle.dump(loss_history, open(os.path.join(args.output_dir, args.run_idx + '_loss_history.pkl'), 'wb'))
        pickle.dump(training_time,
                    open(os.path.join(args.output_dir, args.run_idx + '_time_training.pkl'), 'wb'))
        pickle.dump(validation_time,
                    open(os.path.join(args.output_dir, args.run_idx + '_time_validation.pkl'), 'wb'))
        if current_epoch == 0 or (validation_history[-1] > max(validation_history[:-1])):
            print('\tSaving model...\n')
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            torch.save(model, os.path.join(args.output_dir, args.run_idx + '_model.pt'))
        if current_epoch > args.patience:
            if max(validation_history[:-args.patience]) > max(validation_history[-args.patience:]):
                break


def infer_network(x_test: np.ndarray, y_test: np.ndarray, args: argparse.Namespace) -> None:
    """
    Conduct inference on the trained model.
    :param x_test: Samples for testing.
    :param y_test: Labels for testing.
    :param args: Parsed arguments.
    :return: None.
    """
    model = torch.load(os.path.join(args.output_dir, args.run_idx + '_model.pt'))
    testing_accuracies = []
    for param in model.parameters():
        param.requires_grad = False
    print('\tTesting:')
    begin = time.time()
    x_test_list = [x_test[i:i + args.batch_size] for i in range(0, len(x_test), args.batch_size)]
    y_test_list = [y_test[i:i + args.batch_size] for i in range(0, len(y_test), args.batch_size)]
    heatmaps_per_class = [[] for _ in range(y_test.max() + 1)]
    for x, y in tqdm(zip(x_test_list, y_test_list), total=len(x_test_list)):
        x = torch.from_numpy(x.astype('float32')).unsqueeze(1).type(torch.cuda.FloatTensor)
        y = torch.from_numpy(y.astype('int32')).type(torch.cuda.LongTensor)
        accuracy = (torch.argmax(model(x), dim=1) == y).sum().type(torch.cuda.DoubleTensor) / y.shape[0]
        heatmaps = model.get_heatmaps(x.shape[-1])
        for i in range(y.shape[0]):
            heatmaps_per_class[y[i]].extend([h[i] for h in heatmaps])
        testing_accuracies.append(accuracy.cpu().numpy())
    for i in range(len(heatmaps_per_class)):
        heatmaps_per_class[i] = np.mean(np.array(heatmaps_per_class[i]), axis=0)[0]
    testing_time = time.time() - begin
    testing_accuracy = [np.mean(testing_accuracies)]
    print('\tTesting accuracy: {0}% '.format(testing_accuracy[0]))
    pickle.dump(testing_accuracy,
                open(os.path.join(args.output_dir, args.run_idx + '_testing_accuracy.pkl'), 'wb'))
    pickle.dump(testing_time, open(os.path.join(args.output_dir, args.run_idx + '_time_testing.pkl'), 'wb'))
    pickle.dump(heatmaps_per_class,
                open(os.path.join(args.output_dir, args.run_idx + '_attention_bands.pkl'), 'wb'))


def load_model(n_attention_modules: int, n_classes: int, input_dimension: int):
    """
    Load attention-based model architectures.
    :param n_attention_modules: Number of attention modules = {2, 3, 4}.
    :param n_classes: Number of classes for the problem.
    :param input_dimension: Number of channels that the pixel spectrum contains.
    :return: Model.
    """
    if n_attention_modules == 2:
        print('Model with 2 attention modules.')
        return Model2(n_classes, input_dimension)
    if n_attention_modules == 3:
        print('Model with 3 attention modules.')
        return Model3(n_classes, input_dimension)
    if n_attention_modules == 4:
        print('Model with 4 attention modules.')
        return Model4(n_classes, input_dimension)
    raise Exception


def run(args: argparse.Namespace, selected_bands: np.ndarray = None) -> None:
    """
    Method for running the experiments.
    :param args: Parsed arguments.
    :param selected_bands: Bands selected by the outlier detection algorithm.
    :return: None.
    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("Training attention model for dataset: {}".format(os.path.basename(os.path.normpath(args.dataset_path))))
    samples, labels = get_loader_function(args.dataset_path, args.labels_path)
    if selected_bands is not None:
        samples = samples[:, selected_bands]
    print('Spectral size: {}'.format(samples.shape[-1]))
    (x_train, y_train), \
    (x_val, y_val), \
    (x_test, y_test) = produce_splits(samples,
                                      labels,
                                      args.validation,
                                      args.test)
    model = load_model(args.modules, y_train.max() + 1, x_train.shape[-1])
    model.to(device)
    model.with_attention(args.uses_attention)
    train_network(x_train,
                  y_train,
                  x_val,
                  y_val,
                  model,
                  args)
    infer_network(x_test, y_test, args)


def eval_heatmaps(args: argparse.Namespace) -> np.ndarray:
    """
    Detect outliers in the previous collected heatmaps over all classes.
    :param args: Parsed arguments.
    :return: Array containing selected bands.
    """
    heatmaps = pickle.load(open(os.path.join(args.output_dir, args.run_idx + '_attention_bands.pkl'), 'rb'))
    plot_heatmaps(np.asarray(heatmaps))
    clf = EllipticEnvelope(contamination=float(args.cont))
    outliers = np.asarray(
        [(clf.fit(np.expand_dims(map_, axis=1))).predict(np.expand_dims(map_, axis=1)) for map_ in heatmaps])
    outliers[outliers == 1] = 0
    selected_bands = np.unique(np.asarray([np.nonzero(outlier) for outlier in outliers]))
    print('Selected bands: {0}'.format(selected_bands))
    return selected_bands


if __name__ == "__main__":
    args = arguments()
    run(args)
    selected_bands = eval_heatmaps(args)
    run(args, selected_bands)
