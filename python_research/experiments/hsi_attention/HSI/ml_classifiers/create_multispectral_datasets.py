import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from datasets.generate_trained_models import produce_splits
from selected_bands.load_reduced_models import load_reduced_dataset
from ml_classifiers.utils import arguments, load_classifier, \
    change_data_formatting, get_parameters, get_accuracy_per_class, \
    run_crossvalidation

# OUTPUT_PATH = "C:\\Users\\mmarcinkiewicz\\Desktop\\attention\\trunk\\HSI\\ml_classifiers\\artifacts"
OUTPUT_PATH = r"C:\Users\mmarcinkiewicz\Desktop\attention\trunk\HSI\multispectral_datasets"
np.random.seed(0)


def run():
    samples, labels = load_reduced_dataset(arguments().dataset, arguments().version)
    samples = np.array(samples)
    for k in [1, 4, 8, 16, 32]:
        out_samples = list()
        for i in range(0, samples.shape[-1], k):
            band = np.random.randint(0, min(k, samples.shape[-1]-i), 1)[0]
            out_samples.append(samples[:, :, i + band])
            # out_samples.append(samples[:, :, i:i+k].mean(axis=-1))

        out_samples = np.array(out_samples).transpose((1, 2, 0))
        out_samples = list(out_samples)

        with open(os.path.join(OUTPUT_PATH, arguments().dataset+"_rnd_"+str(k)+".pkl"), "wb") as file:
            pickle.dump((out_samples, labels), file)

        # out_samples = list(out_samples)
        # (x_train, y_train), \
        # (x_val, y_val), \
        # (x_test, y_test) = produce_splits(out_samples,
        #                                   labels,
        #                                   arguments().validation_proportion,
        #                                   arguments().test_proportion)
        #
        # x_train, y_train = change_data_formatting(x_train, y_train)
        # x_val, y_val = change_data_formatting(x_val, y_val)
        # x_test, y_test = change_data_formatting(x_test, y_test)
        #
        # scaler = MinMaxScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_val = scaler.transform(x_val)
        # x_test = scaler.transform(x_test)
        #
        # clf = load_classifier(arguments().classifier)
        # parameters = get_parameters(arguments().classifier)
        #
        # clf = run_crossvalidation(clf, parameters, x_train, y_train, x_val, y_val)
        # clf.fit(x_train, y_train)
        # train_score = clf.score(x_train, y_train)
        # val_score = clf.score(x_val, y_val)
        # test_score = clf.score(x_test, y_test)
        # print(k, round(train_score, 3), round(val_score, 3), round(test_score, 3))

if __name__ == "__main__":
    run()
