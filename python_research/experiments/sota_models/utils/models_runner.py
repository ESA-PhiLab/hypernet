import os
import pickle
import time
from typing import NamedTuple

import torch
from torch.utils.data.dataloader import DataLoader


class History(NamedTuple):
    acc: list
    loss: list
    time: list


class HistoryPack(NamedTuple):
    train: History
    val: History
    test: History


def run_model(args, model, data_prep_function) -> HistoryPack:
    """
    Train, validate and test model.

    :param args: Parsed arguments.
    :param model: Model designed for training, validation and testing.
    :param data_prep_function: Data preparation function.
    :return: Artifacts of the experiment.
    """
    train_dataset, val_dataset, test_dataset = data_prep_function(args=args)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=False, drop_last=True,
                                   pin_memory=True)
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=args.batch, shuffle=False, drop_last=True,
                                 pin_memory=True)

    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False, drop_last=True,
                                  pin_memory=True)

    if args.cont is not None:
        cont = os.path.basename(os.path.normpath(args.cont))
        if cont.endswith('.txt'):
            cont = cont[:-4]
        cont_suffix = '_cont_{}'.format(cont)
    else:
        cont_suffix = ''

    path = os.path.join(
        args.dest_path,
        '{}_run_{}{}'.format(
            args.data_name,
            args.run_idx,
            cont_suffix
        )
    )
    os.makedirs(path, exist_ok=True)

    train_history, val_history = train_model(model=model, train_data_loader=train_data_loader,
                                             val_data_loader=val_data_loader, path=path, args=args)

    test_history = infer_model(model=torch.load(os.path.join(path, "saved_model")),
                               test_data_loader=test_data_loader,
                               path=path)

    return HistoryPack(
        train=train_history,
        val=val_history,
        test=test_history
    )


def infer_model(model, test_data_loader, path) -> History:
    """
    Test previously loaded model.

    :param model: Model for inference.
    :param test_data_loader: Testing data loader.
    :param path: Destination path.
    :return: Inference results.
    """
    test_acc_history = []
    test_loss_history = []
    test_time_history = []
    model.eval()
    begin = time.time()
    for x, y in test_data_loader:
        x = x.type(model.dtype[0])
        y = y.type(model.dtype[1])
        model(x, y, test=True)
    epoch_acc, epoch_loss = model.get_test_results()
    print("Testing -> Accuracy: {} Loss: {}".format(epoch_acc, epoch_loss))
    test_acc_history.append(epoch_acc), test_loss_history.append(epoch_loss)
    test_time_history.append(time.time() - begin)

    print("Saving test results...")
    pickle.dump(test_acc_history, open(os.path.join(path, "test_acc"), "wb"))
    pickle.dump(test_loss_history, open(os.path.join(path, "test_loss"), "wb"))
    pickle.dump(test_time_history, open(os.path.join(path, "test_time"), "wb"))
    pickle.dump(model.acc_per_class, open(os.path.join(path, "acc_per_class"), "wb"))

    return History(
        acc=test_acc_history,
        loss=test_loss_history,
        time=test_time_history
    )


def train_model(model, train_data_loader, val_data_loader, args, path) -> tuple:
    """
    Train model using passed data loaders.

    :param model: Model designed for training.
    :param train_data_loader: Training data loader.
    :param val_data_loader: Validation data loader.
    :param args: Parsed arguments.
    :param path: Destination path.
    :return: History of training.
    """
    train_acc_history = []
    train_loss_history = []
    train_time_history = []
    val_acc_history = []
    val_loss_history = []
    val_time_history = []

    for epoch in range(int(args.epochs)):
        model.train()
        begin = time.time()
        for x, y in train_data_loader:
            x = x.type(model.dtype[0])
            y = y.type(model.dtype[1])
            model.optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            model.optimizer.step()
        epoch_acc, epoch_loss = model.get_train_results()
        print("Training -> Epoch: {} Accuracy: {} Loss: {}".format(epoch, epoch_acc, epoch_loss))
        train_acc_history.append(epoch_acc), train_loss_history.append(epoch_loss)
        train_time_history.append(time.time() - begin)

        model.eval()
        begin = time.time()
        for x, y in val_data_loader:
            x = x.type(model.dtype[0])
            y = y.type(model.dtype[1])
            model(x, y, val=True)
        epoch_acc, epoch_loss = model.get_val_results()
        end = time.time()
        print("Validation -> Epoch: {} Accuracy: {} Loss: {}".format(epoch, epoch_acc, epoch_loss))

        if len(val_acc_history) == 0 or epoch_acc > max(val_acc_history):
            print("Saving improvement...")
            torch.save(model, os.path.join(path, "saved_model"))

        val_acc_history.append(epoch_acc), val_loss_history.append(epoch_loss), val_time_history.append(end - begin)

        pickle.dump(train_acc_history, open(os.path.join(path, "train_acc"), "wb"))
        pickle.dump(train_loss_history, open(os.path.join(path, "train_loss"), "wb"))
        pickle.dump(train_time_history, open(os.path.join(path, "train_time"), "wb"))
        pickle.dump(val_acc_history, open(os.path.join(path, "val_acc"), "wb"))
        pickle.dump(val_loss_history, open(os.path.join(path, "val_loss"), "wb"))
        pickle.dump(val_time_history, open(os.path.join(path, "val_time"), "wb"))

        if epoch > int(args.patience):
            if max(val_acc_history[:-int(args.patience)]) > max(val_acc_history[-int(args.patience):]):
                break

    return (
        History(
            acc=train_acc_history,
            loss=train_loss_history,
            time=train_time_history
        ),
        History(
            acc=val_acc_history,
            loss=val_loss_history,
            time=val_time_history
        )
    )
