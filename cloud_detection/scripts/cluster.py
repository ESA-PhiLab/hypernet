"""
Run K-Means and Gaussian Mixture clustering for provided data
and evaluate the performance through the use of unsupervised metrics.

If you plan on using this implementation, please cite our work:
@INPROCEEDINGS{Grabowski2021IGARSS,
author={Grabowski, Bartosz and Ziaja, Maciej and Kawulok, Michal
and Nalepa, Jakub},
booktitle={IGARSS 2021 - 2021 IEEE International Geoscience
and Remote Sensing Symposium},
title={Towards Robust Cloud Detection in
Satellite Images Using U-Nets},
year={2021},
note={in press}}
"""

import json
import os
import shutil
from pathlib import Path

import clize
import mlflow
import numpy as np
import spectral.io.envi as envi
import yaml
from PIL import Image
from skimage import img_as_ubyte
from skimage.color import label2rgb
from skimage.io import imsave
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from tensorflow.keras.preprocessing.image import load_img

from ml_intuition.data.loggers import log_params_to_mlflow, log_tags_to_mlflow

CLUSTERS = {'km': KMeans,
            'gm': GaussianMixture}

METRICS = {'nmi': normalized_mutual_info_score,
           'ars': adjusted_rand_score}

BACKGROUND_LABEL = 0


def run_clustering(*,
                   config_path: str,
                   alg: str,
                   dest_path: str,
                   n_clusters_max: int,
                   use_mlflow: bool,
                   experiment_name: str = None,
                   run_name: str = None):
    """
    Run clustering for data specified in the experiment configuration.

    :param config_path: Path to the configuration experiment file.
        It should be located in "cfg" directory and contain necessary keys,
        which specify the target dataset names and base path.
    :param alg: Type of algorithm to utilize for clustering.
        Possible options are: "km" for K-Means and "gm" for Gaussian Mixture
        Model respectively.
    :param dest_path: Destination path where all results will be stored.
    :param n_clusters_max: Maximum number of clusters to run on
        all images. Each will be clustered with the number of centres in range
        from two to "n_clusters_max".
    :param use_mlflow: Boolean indicating whether to utilize mlflow.
    :param experiment_name: Name of the experiment. Used only if
        use_mlflow = True.
    :param run_name: Name of the run. Used only if use_mlflow = True.
    """
    if use_mlflow:
        args = locals()
        mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        log_params_to_mlflow(args)
        log_tags_to_mlflow(args['run_name'])

    with open(config_path, 'r') as setup_file:
        setup = yaml.safe_load(setup_file)

    images = setup['exp_cfg']['train_ids'] + setup['exp_cfg']['test_ids']
    os.makedirs(dest_path, exist_ok=True)
    print('Running clustering...')
    for img_name, img_id in images:
        img_base_path = Path(setup['exp_cfg']['dpath']) / img_name / img_id

        gt = envi.open(list(img_base_path.glob("*_fixedmask.hdr"))[0])
        gt = np.array(gt.open_memmap(), dtype=np.int)
        gt = np.where(gt > 128, 1, 0)

        img = np.array(load_img(list(img_base_path.glob('*_B8.TIF'))[0],
                                color_mode='grayscale',
                                target_size=gt.shape))

        mask = np.where(img != BACKGROUND_LABEL)
        data = np.expand_dims(img[mask].ravel(), -1)
        y_true = gt[mask].ravel()

        img_dest_path = os.path.join(dest_path, f'{img_name}-{img_id}')
        os.makedirs(img_dest_path, exist_ok=True)

        metrics = {}
        for n_clusters in range(2, n_clusters_max + 1):
            y_pred = CLUSTERS[alg](n_clusters, random_state=0).fit_predict(
                data)
            metrics[f'{n_clusters}-clusters'] = {
                key: f(labels_true=y_true, labels_pred=y_pred)
                for key, f in METRICS.items()}

            predicted_map = np.full(img.shape, -1)
            predicted_map[mask] = y_pred
            np.savetxt(os.path.join(img_dest_path,
                                    f'{n_clusters}-predicted-map.txt'),
                       predicted_map, fmt='%i')
            imsave(os.path.join(img_dest_path,
                                f'{n_clusters}-predicted-map.png'),
                   img_as_ubyte(label2rgb(predicted_map)))

        gt_map = np.full(img.shape, -1)
        gt_map[mask] = y_true
        np.savetxt(os.path.join(img_dest_path, f'ground-truth-map.txt'),
                   gt_map, fmt='%i')
        imsave(os.path.join(img_dest_path, f'ground-truth-map.png'),
               img_as_ubyte(label2rgb(gt_map)))
        Image.fromarray(img).save(os.path.join(img_dest_path,
                                               'original-map.png'))

        with open(os.path.join(img_dest_path, 'metrics.json'),
                  'w') as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)

        print(metrics)

    if use_mlflow:
        mlflow.log_artifacts(dest_path, artifact_path=dest_path)
        shutil.rmtree(dest_path)


if __name__ == '__main__':
    clize.run(run_clustering)
