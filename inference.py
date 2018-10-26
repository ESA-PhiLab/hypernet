import numpy as np
from scipy.io import loadmat
import os
from keras.models import load_model
from python_research.experiments.utils.datasets.hyperspectral_dataset import HyperspectralDataset
from python_research.experiments.utils.io import load_data, save_to_csv
from keras.utils import to_categorical

def inference(model, dataset):
    labels = to_categorical(dataset.get_labels() - 1, 9)
    data = (dataset.get_data()) / 8000.
    result = model.evaluate(x=data, y=labels)
    acc = result[model.metrics_names.index('acc')]
    return acc


MODELS_DIR = "C:\\Users\mmyller.KPLABS\Documents\datasets\models_for_inference"
DATA_DIR = "C:\\Users\mmyller.KPLABS\Documents\datasets\pavia\\noisy pavia\\new"
OUTPUT_DIR = "C:\\Users\mmyller.KPLABS\Documents\datasets\pavia\\noisy pavia\\results"
GT = load_data("C:\\Users\mmyller.KPLABS\Documents\datasets\pavia\PaviaU_gt.npy")

model_paths = os.listdir(MODELS_DIR)
datasets = os.listdir(DATA_DIR)

for model_path in model_paths:
    model_name = os.path.basename(model_path)
    model = load_model(os.path.join(MODELS_DIR, model_path))
    for data_path in datasets:
        whole_data = load_data(os.path.join(DATA_DIR, data_path))
        for version in range(whole_data.shape[-1]):
            hyp_data = HyperspectralDataset(whole_data[:, :, :, version],
                                            GT)
            acc = inference(model, hyp_data)
            output_path = os.path.join(OUTPUT_DIR, model_name)
            save_to_csv(output_path, [acc])


