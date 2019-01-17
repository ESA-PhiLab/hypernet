import numpy as np

from python_research.experiments.hsi_attention.HSI.datasets.botswana import load_botswana
from python_research.experiments.hsi_attention.HSI.datasets.indian_pines import load_indian_pines
from python_research.experiments.hsi_attention.HSI.datasets.ksc import load_ksc
from python_research.experiments.hsi_attention.HSI.datasets.pavia import load_pavia_university
from python_research.experiments.hsi_attention.HSI.datasets.salinas import load_salinas


def create_sample_label_pairs(samples_by_class):
    all_sample_label_pairs = []

    for idx, class_ in enumerate(samples_by_class):
        for sample in class_:
            labels = np.zeros((1, len(samples_by_class)))
            labels[0][idx] = 1
            all_sample_label_pairs.append((sample, labels.reshape((1, len(samples_by_class)))))

    np.random.shuffle(all_sample_label_pairs)

    samples, labels = zip(*all_sample_label_pairs)

    return np.array(samples), np.array(labels)


def create_class_label_pairs(samples_by_class):
    all_pairs = []

    for idx, class_ in enumerate(samples_by_class):
        small_pairs = []
        for sample in class_:
            labels = np.zeros((1, len(samples_by_class)))
            labels[0][idx] = 1
            small_pairs.append((sample, labels.reshape((1, len(samples_by_class)))))
        all_pairs.append(small_pairs)
    i = 0
    return all_pairs


def produce_splits(X, Y, validation_size, test_size):
    samples_per_class = [[] for _ in range(len(Y[0][0]))]

    for x, y in zip(X, Y):
        samples_per_class[np.argmax(y[0])].append(x)

    del samples_per_class[0]

    lowest_class_population = len(samples_per_class[0])
    for class_ in samples_per_class:
        if len(class_) < lowest_class_population:
            lowest_class_population = len(class_)

    test_set_size = int(lowest_class_population * test_size)

    # Build test set

    test_set = [[] for _ in range(len(samples_per_class))]

    for idx, class_ in enumerate(samples_per_class):
        chosen_indexes = np.random.choice(len(class_), test_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)

        for index in chosen_indexes:
            test_set[idx].append(class_[index])
        samples_per_class[idx] = np.delete(np.array(class_), [chosen_indexes], axis=0)

    # Build test set

    validation_set_size = int(lowest_class_population * validation_size)
    validation_set = [[] for _ in range(len(samples_per_class))]

    for idx, class_ in enumerate(samples_per_class):
        chosen_indexes = np.random.choice(len(class_), validation_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)

        for index in chosen_indexes:
            validation_set[idx].append(class_[index])
        samples_per_class[idx] = np.delete(np.array(class_), [chosen_indexes], axis=0)

    # Build test set

    training_set_size = int(lowest_class_population * (1 - (test_size + validation_size)))
    training_set = [[] for _ in range(len(samples_per_class))]

    for idx, class_ in enumerate(samples_per_class):
        chosen_indexes = np.random.choice(len(class_), training_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)

        for index in chosen_indexes:
            training_set[idx].append(class_[index])
        samples_per_class[idx] = np.delete(np.array(class_), [chosen_indexes], axis=0)

    return create_sample_label_pairs(training_set), \
           create_sample_label_pairs(validation_set), \
           create_sample_label_pairs(test_set)


def get_loader_function(dataset):
    if dataset == 'salinas':
        return load_salinas
    if dataset == 'pavia':
        return load_pavia_university
    if dataset == 'ksc':
        return load_ksc
    if dataset == 'indian_pines':
        return load_indian_pines
    if dataset == 'botswana':
        return load_botswana

    raise Exception


def split_by_batchsize(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        if i + batch_size > len(dataset):
            yield dataset[i:]
        else:
            yield dataset[i:i + batch_size]