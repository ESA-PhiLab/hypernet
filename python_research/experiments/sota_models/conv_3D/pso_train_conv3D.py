import argparse
import copy
import numpy as np
import os
import torch
import pickle
from python_research.experiments.sota_models.conv_3D import conv_3D
from python_research.experiments.sota_models.utils.monte_carlo import prep_monte_carlo
from python_research.experiments.sota_models.utils.models_runner import run_model
from python_research.fastPSO.pso import Pso, Particle, Bounds

from typing import List, NamedTuple


class Arguments(NamedTuple):
    """
    Container for PSO runner arguments
    """
    run_idx: str
    dtype: str
    cont: str
    epochs: int
    data_path: str
    data_name: str
    min_neighborhood_size: int
    max_neighborhood_size: int
    labels_path: str
    batch: int
    patience: int
    dest_path: str
    classes: int
    test_size: float
    val_size: float
    min_channels: List
    max_channels: List
    channels_step: List
    input_depth: int
    swarm_size: int


class PsoRunner:
    """
    PSO runner for 3D convolutional network.
    """

    def __init__(self, args: Arguments):
        """
        Initialization.
        :param args: Arguments for runner.
        """
        self.args = args
        self.archive = {}

    def _extract_parameters(self, position):
        """
        Extracts parameters for particle
        :param position: Position of the particle.
        :return: Particle parameters.
        """
        swarm_neighborhood_size, *swarm_channels = position
        neighborhood_size = int(swarm_neighborhood_size)
        if neighborhood_size % 2 == 0:
            neighborhood_size += 1
        channels = [
            min(int(channel[1]), int(int(channel[0]) + int(channel[2]) * round(channel[3]))) for channel in zip(self.args.min_channels, self.args.max_channels, self.args.channels_step, swarm_channels)
        ]

        return neighborhood_size, channels

    def _objective_function(self, particle: Particle):
        """
        PSO objective function
        :param particle: Particle data.
        :return: Particle score.
        """
        neighborhood_size, channels = self._extract_parameters(particle.position())
        print('Processing: neighborhood = {}, channels = {}'.format(neighborhood_size, channels))

        archive_index = '{},{}'.format(neighborhood_size, channels)
        if archive_index in self.archive:
            return self.archive[archive_index]

        args = lambda: None
        for field in self.args._fields:
            setattr(args, field, getattr(self.args, field))
        args.neighborhood_size = neighborhood_size
        args.input_dim = [args.input_depth, neighborhood_size, neighborhood_size]
        args.channels = channels
        args.run_idx = 'pso_{}_{}'.format(self.args.run_idx, archive_index)

        model = conv_3D.ConvNet3D(
            classes=args.classes,
            channels=list(map(int, args.channels)),
            input_dim=np.asarray(list(map(int, args.input_dim))),
            batch_size=args.batch,
            dtype=args.dtype
        )

        if torch.cuda.is_available():
            model = model.cuda()

        history_pack = run_model(args=args, model=model, data_prep_function=prep_monte_carlo)

        score = max(history_pack.val.acc)
        self.archive[archive_index] = score
        print('Score = {}'.format(score))

        return score

    def run(self):
        """
        Run the optimizer.
        :return: (best_neighborhood, best_channels) tuple.
        """
        min_channels = [0] * len(self.args.min_channels)
        max_channels = [
            int((channel[1] - channel[0]) / channel[2]) + 1 for channel in
                zip(
                    (int(channel) for channel in self.args.min_channels),
                    (int(channel) for channel in self.args.max_channels),
                    (int(channel) for channel in self.args.channels_step)
                )
        ]

        lower_bounds = np.array([self.args.min_neighborhood_size] + min_channels)
        upper_bounds = np.array([self.args.max_neighborhood_size] + max_channels)

        pso = Pso(
            swarm_size=self.args.swarm_size,
            objective_function=self._objective_function,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            threads=1
        )
        best_position, best_score = pso.run()

        best_neighborhood, best_channels = self._extract_parameters(best_position)
        print(
            'Best result: neighborhood = {}, channels = {} (score = {})'.format(
                best_neighborhood,
                best_channels,
                best_score
            )
        )

        if self.args.cont is not None:
            cont = os.path.basename(os.path.normpath(self.args.cont))
            if cont.endswith('.txt'):
                cont = cont[:-4]
            cont_suffix = '_cont_{}'.format(cont)
        else:
            cont_suffix = ''

        path = os.path.join(
            self.args.dest_path,
            '{}_run_pso_{}{}'.format(
                self.args.data_name,
                self.args.run_idx,
                cont_suffix
            )
        )
        os.makedirs(path, exist_ok=True)

        pickle.dump([best_neighborhood, best_channels], open(os.path.join(path, 'best'), 'wb'))

        return (best_neighborhood, best_channels)


def arguments() -> Arguments:
    """
    Arguments parser.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments for runner.')
    parser.add_argument('--run_idx', dest='run_idx', help='Run index.', required=True)
    parser.add_argument('--dtype', dest='dtype', help='Data type used by the model.', required=True)
    parser.add_argument('--cont', dest='cont', help='Path to file containing indexes of selected bands.', type=str)
    parser.add_argument('--epochs', dest='epochs', help='Number of epochs.', type=int, required=True)
    parser.add_argument('--data_path', dest='data_path', help='Path to the data set.', required=True)
    parser.add_argument('--data_name', dest='data_name', help='Name of the data set.', required=True)
    parser.add_argument('--min_neighborhood_size', dest='min_neighborhood_size', help='Min spatial size of the patch.', type=int, required=True)
    parser.add_argument('--max_neighborhood_size', dest='max_neighborhood_size', help='Max spatial size of the patch.', type=int, required=True)
    parser.add_argument('--labels_path', dest='labels_path', help='Path to labels.', required=True)
    parser.add_argument('--batch', dest='batch', help='Batch size.', type=int, required=True)
    parser.add_argument('--patience', dest='patience', help='Number of epochs without improvement.', required=True)
    parser.add_argument('--dest_path', dest='dest_path', help='Destination of the artifacts folder.', required=True)
    parser.add_argument('--classes', dest='classes', help='Number of classes.', type=int, required=True)
    parser.add_argument('--test_size', dest='test_size', help='Test size.', type=float, required=True)
    parser.add_argument('--val_size', dest='val_size', help='Validation size.', type=float, required=True)
    parser.add_argument('--min_channels', dest='min_channels', nargs='+', help='List of min channels.', required=True)
    parser.add_argument('--max_channels', dest='max_channels', nargs='+', help='List of max channels.', required=True)
    parser.add_argument('--channels_step', dest='channels_step', nargs='+', help='List of step value between min and max channels.', required=True)
    parser.add_argument('--input_depth', dest='input_depth', help='Input depth dimensionality.', type=int, required=True)
    parser.add_argument('--swarm_size', dest='swarm_size', help='Swarm size.', type=int, required=True)
    args = vars(parser.parse_args())
    return Arguments(**args)


def main():
    args = arguments()
    pso_runner = PsoRunner(args)
    pso_runner.run()


if __name__ == '__main__':
    main()
