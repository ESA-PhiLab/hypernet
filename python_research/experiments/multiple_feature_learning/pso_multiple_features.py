from train_multiple_features import build_training_set
from python_research.fastPSO.pso import Pso, Particle, Bounds
from keras.callbacks import EarlyStopping
import numpy as np
import os
import argparse


class MultipleFeaturesPso:
    def __init__(
        self,
        original_path,
        gt_path,
        area_path,
        stddev_path,
        diagonal_path,
        moment_path,
        patience
    ):
        self.original_path = original_path
        self.gt_path = gt_path
        self.area_path = area_path
        self.stddev_path = stddev_path
        self.diagonal_path = diagonal_path
        self.moment_path = moment_path
        self.patience = patience
        self.archive = {}

    def run(
        self,
        swarm_size,
        min_batch_size,
        max_batch_size,
        min_nb_samples,
        max_nb_samples,
        min_neighborhood,
        max_neighborhood
    ):
        if min_nb_samples < 10:
            raise ValueError('min_nb_samples must greater or equal to 10')
        if max_nb_samples < 10:
            raise ValueError('max_nb_samples must greater or equal to 10')
        if min_neighborhood <= 0:
            raise ValueError('min_neighborhood must be positive')
        if max_neighborhood <= 0:
            raise ValueError('max_neighborhood must be positive')
        if min_neighborhood % 2 == 0:
            raise ValueError('min_neighborhood must be odd')
        if max_neighborhood % 2 == 0:
            raise ValueError('max_neighborhood must be odd')

        lower_bounds = np.array([min_batch_size, min_nb_samples, min_neighborhood])
        upper_bounds = np.array([max_batch_size, max_nb_samples, max_neighborhood])

        pso = Pso(
            swarm_size=swarm_size,
            objective_function=self._objective_function,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            threads=1
        )
        best_position, best_score = pso.run()

        batch_size, nb_samples, neighborhood = self._extract_parameters(best_position)
        print(
            'Best result: batch size = {}, samples = {}, neighborhood = {} (score = {})'.format(
                batch_size,
                nb_samples,
                neighborhood,
                best_score
            )
        )

    def _objective_function(self, particle: Particle):
        batch_size, nb_samples, neighborhood = self._extract_parameters(particle.position())

        print(
            'Processing: batch size = {}, samples = {}, neighborhood = {}'.format(
                batch_size,
                nb_samples,
                neighborhood
            )
        )

        archive_index = '{}_{}_{}'.format(batch_size, nb_samples, neighborhood)
        if archive_index in self.archive:
            return 1 - self.archive[archive_index]['val_acc'][-1]

        training_set = build_training_set(
            self.original_path,
            self.gt_path,
            self.area_path,
            self.stddev_path,
            self.diagonal_path,
            self.moment_path,
            nb_samples,
            (neighborhood, neighborhood)
        )

        early = EarlyStopping(patience=self.patience)
        history = training_set.model.fit(
            x=training_set.x_train,
            y=training_set.y_train,
            validation_data=(training_set.x_val, training_set.y_val),
            epochs=200,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                early
            ]
        )

        history.history['eval'] = training_set.model.evaluate(
            training_set.x_test,
            training_set.y_test,
            verbose=1
        )[1]

        self.archive[archive_index] = history.history

        score = 1 - history.history['val_acc'][-1]
        print('Score = {}'.format(score))

        return score

    def _extract_parameters(self, position):
        batch_size, nb_samples, neighborhood = position
        batch_size = int(batch_size)
        nb_samples = int(nb_samples)
        neighborhood = int(neighborhood)
        if neighborhood % 2 == 0:
            neighborhood += 1
        return batch_size, nb_samples, neighborhood


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for Multiple Feature Learning'
    )
    parser.add_argument(
        '-o',
        action='store',
        dest='original_path',
        type=str,
        help='Path to the original dataset in .npy format'
    )
    parser.add_argument(
        '-a',
        action='store',
        dest='area_path',
        type=str,
        help='Path to the EAP dataset for area attribute in .npy format'
    )
    parser.add_argument(
        '-s',
        action='store',
        dest='stddev_path',
        type=str,
        help='Path to the EAP dataset for standard deviation attribute in .npy format'
    )
    parser.add_argument(
        '-d',
        action='store',
        dest='diagonal_path',
        type=str,
        help='Path to the EAP dataset for diagonal attribute in .npy format'
    )
    parser.add_argument(
        '-m',
        action='store',
        dest='moment_path',
        type=str,
        help='Path to the EAP dataset for moment attribute in .npy format'
    )
    parser.add_argument(
        '-t',
        action='store',
        dest='gt_path',
        type=str,
        help='Path to the ground truth file in .npy format'
    )
    parser.add_argument(
        '-p',
        action='store',
        dest='patience',
        type=int,
        help='Number of epochs without improvement on validation score before stopping the learning'
    )
    parser.add_argument(
        'swarm',
        action='store',
        type=int,
        help='Swarm size'
    )
    parser.add_argument(
        'minBatchSize',
        action='store',
        type=int,
        help='Minimal size of training batch'
    )
    parser.add_argument(
        'maxBatchSize',
        action='store',
        type=int,
        help='Maximal size of training batch'
    )
    parser.add_argument(
        'minSamples',
        action='store',
        type=int,
        help='Minimal number of training samples used'
    )
    parser.add_argument(
        'maxSamples',
        action='store',
        type=int,
        help='Maximal number of training samples used'
    )
    parser.add_argument(
        'minneighborhood',
        action='store',
        type=int,
        help='Minimal neighborhood size of the pixel'
    )
    parser.add_argument(
        'maxneighborhood',
        action='store',
        type=int,
        help='Maximal neighborhood size of the pixel'
    )

    args = parser.parse_args()

    pso = MultipleFeaturesPso(
        args.original_path,
        args.gt_path,
        args.area_path,
        args.stddev_path,
        args.diagonal_path,
        args.moment_path,
        args.patience
    )

    pso.run(
        args.swarm,
        args.minBatchSize,
        args.maxBatchSize,
        args.minSamples,
        args.maxSamples,
        args.minneighborhood,
        args.maxneighborhood
    )
