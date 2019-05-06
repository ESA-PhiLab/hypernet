from random import shuffle

import numpy as np

from python_research.experiments.sota_models.utils.sets_prep import generate_samples, prep_dataset, unravel_dataset


def prep_monte_carlo(args) -> tuple:
    """
    Finds the size of the smallest population among all classes,
    then divides on three sets:

    - Training set takes (1 - (args.val_size + args.test_size) * lowest_class_population) samples.
    - Validation set takes (args.val_size * lowest_class_population) samples.
    args.val_size is the fraction of samples designed for validation set.
    - Testing set takes (args.test_size * lowest_class_population) samples.
    args.test_size is the fraction of samples designed for testing set.
    :param args: Parsed arguments.
    :return: Training, Validation and Testing objects.
    """
    print("Monte Carlo data prep:")
    samples = generate_samples(args=args)
    samples_by_classes = [[] for _ in range(args.classes)]
    for x in samples:
        samples_by_classes[x[1]].append(x[0].transpose())

    [shuffle(x) for x in samples_by_classes]

    lowest_class_population = len(samples_by_classes[0])
    for class_ in samples_by_classes:
        if len(class_) < lowest_class_population:
            lowest_class_population = len(class_)

    test_set_size = int(lowest_class_population * args.test_size)
    test_set = [[] for _ in range(args.classes)]
    for idx, class_ in enumerate(samples_by_classes):
        chosen_indexes = np.random.choice(len(class_), test_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)
        for index in chosen_indexes:
            test_set[idx].append([class_[index], idx])
        samples_by_classes[idx] = np.delete(np.asarray(class_), [chosen_indexes], axis=0)

    val_set_size = int(lowest_class_population * args.val_size)
    val_set = [[] for _ in range(args.classes)]
    for idx, class_ in enumerate(samples_by_classes):
        chosen_indexes = np.random.choice(len(class_), val_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)
        for index in chosen_indexes:
            val_set[idx].append([class_[index], idx])
        samples_by_classes[idx] = np.delete(np.asarray(class_), [chosen_indexes], axis=0)

    train_set_size = int(lowest_class_population * (1 - (args.val_size + args.test_size)))
    train_set = [[] for _ in range(args.classes)]
    for idx, class_ in enumerate(samples_by_classes):
        chosen_indexes = np.random.choice(len(class_), train_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)
        for index in chosen_indexes:
            train_set[idx].append([class_[index], idx])
        samples_by_classes[idx] = np.delete(np.asarray(class_), [chosen_indexes], axis=0)
    train_set, val_set, test_set = unravel_dataset(train_set=train_set, val_set=val_set, test_set=test_set)
    return prep_dataset(train_set=train_set, val_set=val_set, test_set=test_set)
