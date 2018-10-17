import re
import os
from python_research.experiments.utils import Dataset
from python_research.experiments.utils import PatchData


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def load_patches(directory, classes_count, neighbourhood):
    patch_files = [x for x in sorted_alphanumeric(os.listdir(directory)) if
                   'patch' in x and 'gt' not in x and x.endswith(".npy")]
    gt_files = [x for x in sorted_alphanumeric(os.listdir(directory)) if
                'patch' in x and 'gt' in x and x.endswith(".npy")]
    test_paths = [x for x in os.listdir(directory) if 'test' in x and x.endswith(".npy")]
    train_val_data = PatchData(os.path.join(directory, patch_files[0]),
                           os.path.join(directory, gt_files[0]), neighbourhood)
    for file in range(1, len(patch_files)):
        train_val_data += PatchData(os.path.join(directory, patch_files[file]),
                                os.path.join(directory, gt_files[file]), neighbourhood)
    test_data = Dataset(os.path.join(directory, test_paths[0]),
                        os.path.join(directory, test_paths[1]),
                        0, neighbourhood, classes_count=classes_count,
                        normalize=False, val_split=False)
    train_val_data.train_val_split()
    return train_val_data, test_data


def combine_patches(patches, patches_gt, test, test_gt, neighbourhood, classes_count):
    from python_research.experiments.utils import PatchData
    from python_research.experiments.utils import Dataset
    train_val_data = PatchData(patches[0], patches_gt[0], neighbourhood)
    patches.pop(0)
    patches_gt.pop(0)
    for i in range(0, len(patches)):
        train_val_data += PatchData(patches[i], patches_gt[i], neighbourhood)
    test_data = Dataset(test, test_gt, 0, neighbourhood, classes_count=classes_count,
                        normalize=False, val_split=False)
    train_val_data.train_val_split()
    return train_val_data, test_data