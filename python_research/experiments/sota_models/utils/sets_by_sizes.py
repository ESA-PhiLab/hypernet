from random import shuffle

from python_research.experiments.sota_models.utils.sets_prep import generate_samples, prep_dataset, unravel_dataset


def prep_sets_by_sizes(args) -> tuple:
    """
    Create sets grouped by the number of samples per class.

    args.train_size is the number of samples designed for training per each class.
    args.val_size is the number of samples designed for validation per each class.
    Testing set will contain lowest_class_population - (args.train_size + args.val_size) samples per each class.
    :param args: Parsed arguments containing sizes of datasets.
    :return: Training, Validation and Testing objects.
    """
    print("Balanced data preparation:")
    samples = generate_samples(args=args)
    samples_by_classes = [[] for _ in range(args.classes)]

    for x in samples:
        samples_by_classes[x[1]].append([x[0].transpose(), x[1]])

    [shuffle(x) for x in samples_by_classes]

    lowest_class_population = len(samples_by_classes[0])
    for class_ in samples_by_classes:
        if len(class_) < lowest_class_population:
            lowest_class_population = len(class_)

    train_set = [x[:args.train_size] for x in samples_by_classes]
    val_set = [x[args.train_size:args.train_size + args.val_size] for x in samples_by_classes]
    test_set = [x[args.train_size + args.val_size:lowest_class_population] for x in samples_by_classes]

    train_set, val_set, test_set = unravel_dataset(train_set=train_set, val_set=val_set, test_set=test_set)

    return prep_dataset(train_set=train_set, val_set=val_set, test_set=test_set)
