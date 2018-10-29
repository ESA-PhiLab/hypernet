"""
Fast parallel PSO module
By Pablo Ribalta
https://github.com/pribalta/fastPSO
"""
from typing import List, Tuple
from multiprocessing import Pool
import logging
import datetime
import os
import pickle

import numpy as np


class Logger(object):
    def __init__(self, verbose=True):
        """
        Initialize a logger object if verbosity enabled
        :param verbose: Should log or not
        """
        self._root_logger = None
        self._timestamp = datetime.datetime.now()

        if verbose:
            formatter = logging.Formatter(
                "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
            )
            self._root_logger = logging.getLogger()

            filename = "{}{}{}_{}{}{}_pso.log".format(*self.timestamp())
            file_handler = logging.FileHandler(
                "{0}/{1}.log".format(os.path.realpath(__file__), filename)
            )
            file_handler.setFormatter(formatter)
            self._root_logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self._root_logger.addHandler(console_handler)

    def log(self, message: str, error: bool=False) -> None:
        """
        Log message
        :param message: string to log
        :param error: if message should be logged as error
        :return: None
        """
        if self._root_logger:
            self._root_logger.log(logging.CRITICAL if error else logging.INFO, message)

        if error:
            raise ValueError(message)

    def timestamp(self) -> Tuple:
        return (
            self._timestamp.year,
            self._timestamp.month,
            self._timestamp.day,
            self._timestamp.hour,
            self._timestamp.min,
            self._timestamp.second
        )


class ObjectiveFunctionBase(object):
    """
    Objective function base class to be overriden
    """

    def __call__(self, *args, **kwargs) -> float:
        """
        Function to be evaluated
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class Bounds(object):
    """
    Encapsulation of PSO bounds. Ensures creation and validation
    """

    def __init__(
        self,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        logger: Logger = Logger(verbose=False)
    ):
        """
        Type encapsulating the bounds of PSO
        :param lower_bound: maximum values for parameters
        :param upper_bound: minimum values for parameters
        """
        self._logger = logger

        if len(lower_bound.shape) > 1 != len(upper_bound.shape) > 1:
            self._logger.log(
                "Lower and upper bound must have 1D."
                " Received {} and {}".format(len(lower_bound.shape), len(upper_bound.shape)),
                error=True
            )

        if lower_bound.shape != upper_bound.shape:
            self._logger.log("Lower and upper bound must have the same shape."
                             " Received {} and {}".format(lower_bound.shape, upper_bound.shape),
                             error=True)

        if lower_bound.dtype != upper_bound.dtype:
            self._logger.log("Upper and lower bound must share the same type."
                             " Found {} and {}".format(lower_bound.dtype, upper_bound.dtype),
                             error=True)

        if not np.all(np.greater_equal(upper_bound, lower_bound)):
            self._logger.log("Upper bound values must be greater or equal than lower bound values."
                             " Received {} and {}".format(upper_bound, lower_bound),
                             error=True)

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        self._logger.log("Initial upper bound: {}".format(upper_bound))
        self._logger.log("Initial lower bound: {}".format(lower_bound))

    def lower(self) -> np.ndarray:
        """
        Return the lower bound
        :return: lower bound
        """
        return self._lower_bound

    def upper(self) -> np.ndarray:
        """
        Return the upper bound
        :return: upper bound
        """
        return self._upper_bound


class PsoParameters(object):
    """
    Encapsulates the parameters for PSO velocity updates
    """

    def __init__(
        self,
        omega: float,
        phip: float,
        phig: float,
        logger: Logger = Logger(verbose=False)
    ):
        """
        Construct a bundle for the PSO parameters
        :param omega: omega coefficient
        :param phip: phip coefficient
        :param phig: phig coefficient
        """
        self._logger = logger

        if omega > 1 or omega < 0:
            self._logger.log("Value for omega should be [0.0, 1.0]", error=True)

        if phip > 1 or phip < 0:
            self._logger.log("Value for phip should be [0.0, 1.0]", error=True)

        if phig > 1 or phig < 0:
            self._logger.log("Value for phig should be [0.0, 1.0]", error=True)

        self._omega = omega
        self._phip = phip
        self._phig = phig

        self._logger.log("Initial omega parameter: {}".format(omega))
        self._logger.log("Initial phip parameter: {}".format(phip))
        self._logger.log("Initial phig parameter: {}".format(phig))

    def omega(self) -> float:
        """
        Get omega
        :return: float
        """
        return self._omega

    def phip(self) -> float:
        """
        Get phip
        :return: float
        """
        return self._phip

    def phig(self) -> float:
        """
        Get phig
        :return: float
        """
        return self._phig


class Particle(object):
    """
    A particle is the key entity in the PSO algorithm
    """

    def __init__(
        self,
        bounds: Bounds,
        parameters: PsoParameters,
        logger: Logger = Logger(verbose=False)
    ):
        """
        Create a particle
        :param bounds: boundaries for particle position
        :param parameters: parameters for particle updates
        """
        self._logger = logger

        self._bounds = bounds
        self._parameters = parameters

        self._position = [self._calculate_initial_position()]
        self._velocity = [self._calculate_initial_velocity()]
        self._score = []

        self._logger.log("Created particle:\n\tPosition: {}\n\tVelocity: {}".format(
            self.position(),
            self.velocity()
        ))

    def position(self) -> np.ndarray:
        """
        Get current position
        :return: np.ndarray
        """
        return self._position[-1]

    def velocity(self) -> np.ndarray:
        """
        Get current velocity
        :return: np.ndarray
        """
        return self._velocity[-1]

    def best_position(self) -> np.ndarray:
        """
        Get the best position in the history of a particle
        :return: np.ndarray
        """
        if len(self._position) != len(self._score):
            self._logger.log("Amount of positions should be the same as amount of scores."
                             "Received {} and {}".format(len(self._position), len(self._score)),
                             error=True)

        return self._position[np.argsort(self._score)[-1]]

    def best_score(self) -> float:
        """
        Get best score in the lifetime of a particle
        :return: float
        """
        if not self._score:
            self._logger.log("Cannot update velocity while scores are empty. Evaluate first.",
                             error=True)

        return self._score[np.argsort(self._score)[-1]]

    def update(self, swarm_best: np.ndarray) -> None:
        """
        Update the velocity and position of a particle
        :param swarm_best:
        :return: None
        """
        if not self._score:
            self._logger.log("Cannot update while scores are empty. Evaluate first.",
                             error=True)

        # pylint: disable = invalid-name
        rp, rg = self._initialize_random_coefficients

        self._velocity.append(self._parameters.omega() * self.velocity()
                              + self._parameters.phip() * rp * (self.best_position()
                                                                - self.position())
                              + self._parameters.phig() * rg * (swarm_best
                                                                - self.position()))
        self._position.append(self._calculate_position())

        self._logger.log("Updated particle:\n\tPosition: {}\n\tVelocity: {}".format(
            self.position(),
            self.velocity()
        ))

    def update_score(self, score: float) -> None:
        """
        Update a particle's score
        :param score: score obtained by a particle
        :return: None
        """
        self._score.append(score)
        self._logger.log("Updated particle:\n\tScore: {}".format(score))

    def _calculate_initial_position(self) -> np.ndarray:
        """
        Initialize particle's position
        :return: np.ndarray
        """
        return np.array([np.random.uniform(low, high)
                         for low, high in zip(self._bounds.lower(),
                                              self._bounds.upper())]) \
            .astype(self._bounds.lower().dtype)

    def _calculate_initial_velocity(self) -> np.ndarray:
        """
        Initialize a particle's velocity
        :return: np.ndarray
        """
        return np.array([np.random.uniform(-(high - low), high - low)
                         for low, high in zip(self._bounds.lower(),
                                              self._bounds.upper())]) \
            .astype(self._bounds.lower().dtype)

    @property
    def _initialize_random_coefficients(self) -> Tuple[float, float]:
        """
        Initialize the random coefficients for the velocity update
        :return: Tuple
        """
        return np.random.uniform(0, 1), np.random.uniform(0, 1)

    def _calculate_position(self) -> np.ndarray:
        """
        Calculate a particle's position
        :return: New particle's position
        """
        new_position = self._position[-1] + self._velocity[-1]

        for i in range(new_position.size):
            if self._bounds.lower()[i] > new_position[i]:
                new_position[i] = self._bounds.lower()[i]
            elif self._bounds.upper()[i] < new_position[i]:
                new_position[i] = self._bounds.upper()[i]

        return new_position

    def last_movement(self) -> float:
        """
        Calculate last movement
        :return: float
        """
        return np.linalg.norm(self._position[-2] - self._position[-1])

    def last_improvement(self) -> float:
        """
        Calculate last improvement
        :return: float
        """
        if not self._score:
            self._logger.log("Cannot calculate improvement while scores are empty. Evaluate first.",
                             error=True)

        if len(self._score) == 1:
            return float("inf")

        return self._score[-1] - self._score[-2]


class Swarm(object):
    """
    Encapsulate and efficiently manage a set of particles
    """

    def __init__(
        self,
        swarm_size: int,
        bounds: Bounds,
        parameters: PsoParameters,
        minimum_step: float,
        minimum_improvement: float,
        objective_function: ObjectiveFunctionBase,
        logger: Logger = Logger(verbose=False)
    ):
        """
        Constructs a swarm
        :param swarm_size: Number of particles
        :param bounds: Bounds for the parameter space
        :param minimum_step: constraint for particle movement
        :param minimum_improvement: constraint for particle improvement
        """
        self._logger = logger
        self._objective_function = objective_function

        if swarm_size <= 0:
            self._logger.log("Swarm size must be greater than zero",
                             error=True)

        self._particles = [Particle(bounds, parameters, logger) for _ in range(swarm_size)]
        for particle in self._particles:
            score = self._objective_function(particle)
            particle.update_score(score)

        self._minimum_step = minimum_step
        self._minimum_improvement = minimum_improvement

        self._logger.log(
            "Created swarm:\n\tsize: {}\n\tMinimum step: {} \n\tMinimum improvement: {}".format(
                swarm_size,
                minimum_step,
                minimum_improvement
            )
        )

    def __iter__(self):
        return self._particles.__iter__()

    def __len__(self):
        return len(self._particles)

    def still_improving(self) -> bool:
        """
        Determie if the swarm is still improving given the minimum improvement constraint
        :return: bool
        """
        improvement_deltas = [particle.last_improvement()
                              for particle in self._particles]

        for delta in improvement_deltas:
            if delta > self._minimum_improvement:
                return True

        return False

    def still_moving(self) -> bool:
        """
        Determie if the swarm is still moving given the minimum step constraint
        :return: bool
        """
        movement_deltas = [particle.last_movement()
                           for particle in self._particles]

        for delta in movement_deltas:
            if delta > self._minimum_step:
                return True

        return False

    def update(self) -> None:
        """
        Update the velocity, position and score of all particles in the swarm
        :return: None
        """
        swarm_best_position = self.best_position()

        for particle in self._particles:
            particle.update(swarm_best_position)
            score = self._objective_function(particle)
            particle.update_score(score)

    def best_position(self) -> np.ndarray:
        """
        Return the best position in the swarm
        :return: np.ndarray
        """
        best_positions = [particle.best_position() for particle in self._particles]
        best_scores = [particle.best_score() for particle in self._particles]

        return best_positions[np.argsort(best_scores)[-1]]

    def best_score(self) -> float:
        """
        Return the best score in the swarm
        :return: float
        """
        best_scores = [particle.best_score() for particle in self._particles]

        return best_scores[np.argsort(best_scores)[-1]]


class Pso(object):
    """
    Pso encapsulates the creation and successive updates of a swarm of particles
    """

    def __init__(
        self,
        objective_function: ObjectiveFunctionBase,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        swarm_size: int,
        omega: float = 0.5,
        phip: float = 0.5,
        phig: float = 0.5,
        maximum_iterations: int = 100,
        minimum_step: float = 10e-8,
        minimum_improvement: float = 10e-8,
        threads: int = 1,
        verbose: bool = False
    ):
        """
        Constructor of a Particle Swarm Optimizer
        :param objective_function: Objective function reporting the score
        :param lower_bound: lower bound for parameters
        :param upper_bound: upper bound for parameters
        :param swarm_size: number of particles in the swarm
        :param omega: omega parameter for velocity updates
        :param phip: phip parameter for velocity updates
        :param phig: phig parameter for velocity updates
        :param maximum_iterations: maximum number of iterations for optimization
        :param minimum_step: minimum particle distance
        :param minimum_improvement: minimum allowed improvement
        :param threads: number of execution threads
        :param verbose: enable to receive information about the progress
        """
        self._logger = Logger(verbose)

        if maximum_iterations <= 0:
            self._logger.log("Maximum number of iterations must be greater than zero", error=True)

        self._maximum_iterations = maximum_iterations

        self._swarm = Swarm(
            swarm_size,
            Bounds(lower_bound, upper_bound,  self._logger),
            PsoParameters(omega, phip, phig,  self._logger),
            minimum_step,
            minimum_improvement,
            objective_function,
            self._logger
        )

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run particle swarm optimization
        :return: (tuple) best position, best score
        """
        for _ in range(self._maximum_iterations):
            self._swarm.update()

            if not self._swarm.still_improving() or not self._swarm.still_moving():
                break

        return self._swarm.best_position(), self._swarm.best_score()
