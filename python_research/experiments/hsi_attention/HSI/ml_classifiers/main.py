import os
from tqdm import tqdm
from time import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from datasets.generate_trained_models import produce_splits
from selected_bands.load_reduced_models import load_reduced_dataset
from multispectral_datasets.load_multispectral_models import load_multispectral_dataset
from ml_classifiers.utils import arguments, load_classifier, \
    change_data_formatting, get_parameters, get_accuracy_per_class, \
    run_crossvalidation

OUTPUT_PATH = "C:\\Users\\mmarcinkiewicz\\Desktop\\attention\\trunk\\HSI\\ml_classifiers\\artifacts\\multispectral"
# OUTPUT_PATH = "\\\\morpheus\\pub-Transfer\\m\\mmarcinkiewicz\\artifacts\\multispectral"


def run():
    report_accuracy = pd.DataFrame(columns=['Tr_acc', 'V_acc', 'Te_acc', 'time'])
    report_accuracy_per_class = pd.DataFrame()
    for run_id in tqdm(range(arguments().runs)):
        samples, labels = load_multispectral_dataset(arguments().dataset, arguments().mode, arguments().version)
        (x_train, y_train), \
        (x_val, y_val), \
        (x_test, y_test) = produce_splits(samples,
                           labels,
                           arguments().validation_proportion,
                           arguments().test_proportion)

        x_train, y_train = change_data_formatting(x_train, y_train)
        x_val, y_val = change_data_formatting(x_val, y_val)
        x_test, y_test = change_data_formatting(x_test, y_test)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        clf = load_classifier(arguments().classifier)
        parameters = get_parameters(arguments().classifier)

        clf = run_crossvalidation(clf, parameters, x_train, y_train, x_val, y_val)
        start_time = time()
        clf.fit(x_train, y_train)
        training_time = time() - start_time
        train_score = clf.score(x_train, y_train)
        val_score = clf.score(x_val, y_val)
        test_score = clf.score(x_test, y_test)

        report = pd.DataFrame([train_score, val_score, test_score, training_time]).T
        report.columns = ['Tr_acc', 'V_acc', 'Te_acc', 'time']
        report_accuracy = report_accuracy.append(report, ignore_index=True)

        # print("Dataset", arguments().dataset, arguments().version,
        #       "\nClassifier:", arguments().classifier,
        #       "; Parameters", best_params,
        #       "\nTrain score", train_score,
        #       "\nValidation score:", val_score,
        #       "\nTest score:", test_score,
        #       "\nTime:", training_time, "s.")

        accuracy_per_class = get_accuracy_per_class(y_test, clf.predict(x_test))
        report_accuracy_per_class['run' + str(run_id)] = accuracy_per_class

    mean = pd.DataFrame(report_accuracy.mean()).T
    mean.columns = ['Tr_acc', 'V_acc', 'Te_acc', 'time']
    report_accuracy = report_accuracy.append(mean, ignore_index=True)
    std = pd.DataFrame(report_accuracy.std()).T
    std.columns = ['Tr_acc', 'V_acc', 'Te_acc', 'time']
    report_accuracy = report_accuracy.append(std, ignore_index=True)
    report_accuracy.index = ["run" + str(item) for item in report_accuracy.index[:-2]] + ['mean', 'std']

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    report_accuracy.to_excel(os.path.join(OUTPUT_PATH, arguments().dataset +
                                          "_" + arguments().mode +
                                          "_" + str(arguments().version) +
                                          "_" + arguments().classifier + "_acc.xlsx"))

    report_accuracy_per_class['mean'] = report_accuracy_per_class.T.mean()
    report_accuracy_per_class['std'] = report_accuracy_per_class.T.std()
    report_accuracy_per_class = report_accuracy_per_class.T
    report_accuracy_per_class.to_excel(os.path.join(OUTPUT_PATH, arguments().dataset +
                                                    "_" + arguments().mode +
                                                    "_" + str(arguments().version) +
                                                    "_" + arguments().classifier + "_acc_pc.xlsx"))


if __name__ == "__main__":
    run()
