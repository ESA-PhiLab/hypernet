import os
import pickle
import tqdm

from datasets.generate_trained_models import get_loader_function


def main():
    suffixes_salinas = [path for path in os.listdir(os.path.join('selected_bands', 'salinas'))]
    suffixes_pavia = [path for path in os.listdir(os.path.join('selected_bands', 'pavia'))]


    for suffix in tqdm.tqdm(suffixes_salinas):
        with open(os.path.join('selected_bands', 'salinas', suffix), 'r') as input:
            bands = [int(band.strip()) for band in input.readlines()]
            samples, labels = get_loader_function('salinas')()

            filtered_samples = [sample[:,bands] for sample in samples]

            pickle.dump((filtered_samples, labels), open(os.path.join('selected_bands', 'salinas', 'reduced_dataset_' + suffix[:-4] + '.pkl'), 'wb'))


    for suffix in tqdm.tqdm(suffixes_pavia):
        with open(os.path.join('selected_bands', 'pavia', suffix), 'r') as input:
            bands = [int(band.strip()) for band in input.readlines()]
            samples, labels = get_loader_function('pavia')()

            filtered_samples = [sample[:,bands] for sample in samples]

            pickle.dump((filtered_samples, labels), open(os.path.join('selected_bands', 'pavia', 'reduced_dataset_' + suffix[:-4] + '.pkl'), 'wb'))

if __name__ == '__main__':
    main()