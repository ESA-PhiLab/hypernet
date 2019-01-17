# for dataset in ['salinas']:
#     for clf in ['svm']:
#         for version in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 1.0]:
#             print("python -m ml_classifiers.main --runs 30 --dataset {} --version {} --validation 0.1 --test 0.1 --clf {}".format(dataset, version, clf))

for dataset in ['pavia']:
    for clf in ['dt']:
        for version in [1, 32]:
            for mode in ['max', 'avg', 'rnd']:
                print("python -m ml_classifiers.main --runs 10 --dataset {} --version {} --validation 0.1 --test 0.1 --clf {} --mode {}".format(dataset, version, clf, mode))
