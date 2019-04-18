import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
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


def train_network(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, model,
                  args: argparse.Namespace):
    """
    Train and validate model.

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
    for current_epoch in range(args.epochs):
        torch.enable_grad()
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        print("Epoch {}:".format(current_epoch))
        print("\tTraining phase:")

        training_accuracies = []
        losses = []
        begin = time.time()

        x_train_list = [x_train[i:i + args.batch_size] for i in range(0, len(x_train), args.batch_size)]
        y_train_list = [y_train[i:i + args.batch_size] for i in range(0, len(y_train), args.batch_size)]

        for x, y in tqdm(zip(x_train_list, y_train_list), total=len(x_train_list)):
            x = torch.from_numpy(x.astype("float32")).unsqueeze(1).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y.astype("int32")).type(torch.cuda.LongTensor)
            model.zero_grad()
            model.optimizer.zero_grad()
            out = model(x, y, infer=False)
            loss = model.loss(out, y)
            losses.append(loss.clone().detach().cpu().numpy())
            accuracy = (torch.argmax(out, dim=1) == y).sum().type(torch.cuda.DoubleTensor) / y.shape[0]
            training_accuracies.append(accuracy.cpu().numpy())
            loss.backward()
            model.optimizer.step()

        training_time.append(time.time() - begin)

        validation_accuracies = []

        torch.no_grad()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        print("\tValidation phase:")
        begin = time.time()

        x_val_list = [x_val[i:i + args.batch_size] for i in range(0, len(x_val), args.batch_size)]
        y_val_list = [y_val[i:i + args.batch_size] for i in range(0, len(y_val), args.batch_size)]

        for x, y in tqdm(zip(x_val_list, y_val_list), total=len(x_val_list)):
            x = torch.from_numpy(x.astype("float32")).unsqueeze(1).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y.astype("int32")).type(torch.cuda.LongTensor)
            accuracy = (torch.argmax(model(x, y, infer=False), dim=1) == y).sum().type(torch.cuda.DoubleTensor) / \
                       y.shape[0]
            validation_accuracies.append(accuracy.cpu().numpy())

        validation_time.append(time.time() - begin)
        loss_history.append(np.mean(losses))
        training_history.append(np.mean(training_accuracies))
        validation_history.append(np.mean(validation_accuracies))
        print("\tLoss: {0}% ".format(loss_history[-1]))
        print("\tTraining accuracy: {0}% ".format(training_history[-1]))
        print("\tValidation accuracy: {0}% ".format(validation_history[-1]))
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        pickle.dump(training_history,
                    open(os.path.join(args.output_dir, args.run_idx + "_training_history.pkl"), "wb"))
        pickle.dump(validation_history,
                    open(os.path.join(args.output_dir, args.run_idx + "_validation_history.pkl"), "wb"))
        pickle.dump(loss_history, open(os.path.join(args.output_dir, args.run_idx + "_loss_history.pkl"), "wb"))
        pickle.dump(training_time,
                    open(os.path.join(args.output_dir, args.run_idx + "_time_training.pkl"), "wb"))
        pickle.dump(validation_time,
                    open(os.path.join(args.output_dir, args.run_idx + "_time_validation.pkl"), "wb"))
        if current_epoch == 0 or (validation_history[-1] > max(validation_history[:-1])):
            print("\tSaving model...")
            torch.save(model, os.path.join(args.output_dir, args.run_idx + "_model.pt"))
        if current_epoch > args.patience:
            if max(validation_history[:-args.patience]) > max(validation_history[-args.patience:]):
                print("\tBail...")
                break


def infer_network(x_test: np.ndarray, y_test: np.ndarray, args: argparse.Namespace, input_size: int) -> None:
    """
    Conduct inference on the trained model.

    :param x_test: Samples for testing.
    :param y_test: Labels for testing.
    :param args: Parsed arguments.
    :param input_size: Size of the initial band spectrum.
    :return: None.
    """
    model = torch.load(os.path.join(args.output_dir, args.run_idx + "_model.pt"))
    testing_accuracies = []

    torch.no_grad()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("\tTesting:")

    begin = time.time()
    x_test_list = [x_test[i:i + args.batch_size] for i in range(0, len(x_test), args.batch_size)]
    y_test_list = [y_test[i:i + args.batch_size] for i in range(0, len(y_test), args.batch_size)]

    for x, y in tqdm(zip(x_test_list, y_test_list), total=len(x_test_list)):
        x = torch.from_numpy(x.astype("float32")).unsqueeze(1).type(torch.cuda.FloatTensor)
        y = torch.from_numpy(y.astype("int32")).type(torch.cuda.LongTensor)
        accuracy = (torch.argmax(model(x, y, infer=True), dim=1) == y).sum().type(torch.cuda.DoubleTensor) / y.shape[0]
        testing_accuracies.append(accuracy.cpu().numpy())

    testing_time = time.time() - begin
    testing_accuracy = [np.mean(testing_accuracies)]
    print("\tTesting accuracy: {}% ".format(testing_accuracy[0]))

    heatmaps_per_class = model.get_heatmaps(input_size=input_size)

    pickle.dump(testing_accuracy,
                open(os.path.join(args.output_dir, args.run_idx + "_testing_accuracy.pkl"), "wb"))
    pickle.dump(testing_time, open(os.path.join(args.output_dir, args.run_idx + "_time_testing.pkl"), "wb"))
    pickle.dump(heatmaps_per_class,
                open(os.path.join(args.output_dir, args.run_idx + "_attention_bands.pkl"), "wb"))


def load_model(n_attention_modules: int, n_classes: int, input_dimension: int):
    """
    Load attention-based model architectures.
    :param n_attention_modules: Number of attention modules = {2, 3, 4}.
    :param n_classes: Number of classes for the problem.
    :param input_dimension: Size of the spectrum channel.
    :return: Instance of the model.
    """

    if n_attention_modules == 2:
        print("Model with 2 attention modules.")
        return Model2(n_classes, input_dimension)
    if n_attention_modules == 3:
        print("Model with 3 attention modules.")
        return Model3(n_classes, input_dimension)
    if n_attention_modules == 4:
        print("Model with 4 attention modules.")
        return Model4(n_classes, input_dimension)


def run(args: argparse.Namespace, selected_bands: np.ndarray = None) -> None:
    """
    Method for running the experiments.
    :param args: Parsed arguments.
    :param selected_bands: Bands selected by the outlier detection algorithm.
    :return: None.
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("Training attention model for dataset: {}".format(os.path.basename(os.path.normpath(args.dataset_path))))
    samples, labels = get_loader_function(data_path=args.dataset_path, ref_map_path=args.labels_path)
    if selected_bands is not None:
        print("Selecting bands...")
        samples = samples[..., selected_bands]
        print("Number of selected bands: {}".format(samples.shape[-1]))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = produce_splits(samples=samples,
                                                                          labels=labels,
                                                                          validation_size=args.validation,
                                                                          test_size=args.test)
    model = load_model(n_attention_modules=args.modules,
                       n_classes=int(y_train.max() + 1),
                       input_dimension=x_train.shape[-1])
    model.to(device)
    if args.attn == "true":
        model.uses_attention = True
    if args.attn == "false":
        model.uses_attention = False
    train_network(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, model=model, args=args)
    infer_network(x_test=x_test, y_test=y_test, args=args, input_size=x_train.shape[-1])


def plot_heatmaps(heatmaps: np.ndarray, args: argparse.Namespace):
    fig, axis = plt.subplots()
    heatmap = axis.pcolor(heatmaps)
    axis.set_yticklabels([str(class_ + 1) for class_ in range(heatmaps.shape[0])], minor=False)
    plt.colorbar(heatmap)
    fig.set_size_inches(12, 4)
    plt.title("Attention heatmaps scores")
    plt.ylabel("Class index")
    plt.xlabel("Band index")
    plt.savefig(os.path.join(args.output_dir, args.run_idx + "_attention_map.pdf"))


def eval_heatmaps(args: argparse.Namespace) -> np.ndarray:
    """
    Detect outliers in the collected heatmaps.

    :param args: Parsed arguments.
    :return: Array containing selected bands.
    """
    heatmaps = pickle.load(open(os.path.join(args.output_dir, args.run_idx + "_attention_bands.pkl"), "rb"))
    plot_heatmaps(heatmaps, args)
    clf = EllipticEnvelope(contamination=float(args.cont))
    outliers = np.asarray(
        [(clf.fit(np.expand_dims(map_, axis=1))).predict(np.expand_dims(map_, axis=1)) for map_ in heatmaps])
    outliers[outliers == 1] = 0
    nonzero = np.asarray([np.nonzero(outlier) for outlier in outliers]).squeeze()
    selected_bands = np.unique(nonzero)
    print("Selected bands: {0}".format(selected_bands))
    return selected_bands


def main():
    args = arguments()
    run(args)
    selected_bands = eval_heatmaps(args)
    run(args, selected_bands)


if __name__ == "__main__":
    main()
