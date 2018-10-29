'''
Dataset segmentation
'''
import math
import random
from typing import List, Sequence
import numpy as np


class Point:
    '''
    Represents a point in 2D space
    '''
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    @staticmethod
    def from_point(other):
        '''
        Creates a point from another point
        '''
        return Point(other.x, other.y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Rect:
    '''
    Represents a rectangle
    '''
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        min_x, max_x = (x1, x2) if x1 < x2 else (x2, x1)
        min_y, max_y = (y1, y2) if y1 < y2 else (y2, y1)
        self.min = Point(min_x, min_y)
        self.max = Point(max_x, max_y)

    @staticmethod
    def from_points(point1: Point, point2: Point):
        '''
        Creates a rectangle from two points
        :param point1: (Point) Point at first corner
        :param point2: (Point) Point at second corner
        '''
        return Rect(point1.x, point1.y, point2.x, point2.y)

    width = property(lambda self: self.max.x - self.min.x)
    height = property(lambda self: self.max.y - self.min.y)

    def __str__(self):
        return "({}, {}, {}, {})".format(self.min.x, self.min.y, self.max.x, self.max.y)


class BoundTree:
    '''
    Represents a tree with 2D bounds
    '''
    def __init__(self, bounds: Rect):
        self.bounds = bounds
        self.sub_trees = []

    def __iter__(self):
        if not self.sub_trees:
            yield self.bounds
        else:
            for sub_tree in self.sub_trees:
                for bounds in sub_tree:
                    yield bounds

    def can_divide(self, must_fit: Rect) -> bool:
        '''
        Checks whether the tree can be divided in any direction
        :param must_fit: (Rect) Rectangle that must fit into each side after the division
        '''
        return (
            self.can_divide_horizontal(must_fit.width) or
            self.can_divide_vertical(must_fit.height)
        )

    def can_divide_horizontal(self, must_fit: int) -> bool:
        '''
        Checks whether the tree can be divided horizontally
        :param must_fit: (int) Width that must fit into each side after the division
        '''
        return self.bounds.width >= 2.0 * must_fit

    def can_divide_vertical(self, must_fit: int) -> bool:
        '''
        Checks whether the tree can be divided vertically
        :param must_fit: (int) Height that must fit into each side after the division
        '''
        return self.bounds.height >= 2.0 * must_fit

    def random_deep_divide(self, must_fit: Rect):
        '''
        Randomly divides the tree and its children until no child can be divided
        :param must_fit: (Rect) Rectangle that must fit into children
        '''
        if not self.can_divide(must_fit):
            return

        self.random_divide(must_fit)
        for sub_tree in self.sub_trees:
            sub_tree.random_deep_divide(must_fit)

    def random_divide(self, must_fit: Rect):
        '''
        Randomly divides the tree horizontally or vertically
        :param must_fit: (Rect) Rectangle that must fit into each side after the division
        '''
        orientations = [
            0 if self.can_divide_horizontal(must_fit.width) else 1,
            1 if self.can_divide_vertical(must_fit.height) else 0
        ]
        orientation = random.choice(orientations)
        if orientation == 0:
            self.random_divide_horizontal(must_fit.width)
        else:
            self.random_divide_vertical(must_fit.height)

    def random_divide_horizontal(self, must_fit: int):
        '''
        Divides the tree horizontally at a random position
        :param must_fit: (int) Width that must fit into each side after the division
        '''
        position = random.randint(must_fit, self.bounds.width - must_fit)
        self.divide_horizontal(self.bounds.min.x + position)

    def random_divide_vertical(self, must_fit: int):
        '''
        Divides the tree vertically at a random position
        :param must_fit: (int) Height that must fit into each side after the division
        '''
        position = random.randint(must_fit, self.bounds.height - must_fit)
        self.divide_vertical(self.bounds.min.y + position)

    def divide_horizontal(self, position: int):
        '''
        Divides the tree horizonstally
        :param position: (int) A position at which to split the tree
        '''
        sub_bounds1 = Rect(self.bounds.min.x, self.bounds.min.y, position, self.bounds.max.y)
        sub_bounds2 = Rect(position, self.bounds.min.y, self.bounds.max.x, self.bounds.max.y)
        self.sub_trees.clear()
        self.sub_trees.append(BoundTree(sub_bounds1))
        self.sub_trees.append(BoundTree(sub_bounds2))

    def divide_vertical(self, position: int):
        '''
        Divides the tree vertically
        :param position: (int) A position at which to split the tree
        '''
        sub_bounds1 = Rect(self.bounds.min.x, self.bounds.min.y, self.bounds.max.x, position)
        sub_bounds2 = Rect(self.bounds.min.x, position, self.bounds.max.x, self.bounds.max.y)
        self.sub_trees.clear()
        self.sub_trees.append(BoundTree(sub_bounds1))
        self.sub_trees.append(BoundTree(sub_bounds2))


def get_bounding_rect(rects: List[Rect]) -> Rect:
    '''
    Calculates a bounding rectangle for other rectangles
    :param rects: (List[Rect]) List of rectangles
    :return: (Rect) Bounding rectangle
    '''
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for rect in rects:
        min_x = min_x if rect.min.x > min_x else rect.min.x
        min_y = min_y if rect.min.y > min_y else rect.min.y
        max_x = max_x if rect.max.x < max_x else rect.max.x
        max_y = max_y if rect.max.y < max_y else rect.max.y

    return Rect(min_x, min_y, max_x, max_y)


def randomize_positions(rects: List[Rect], bounds: Rect) -> Sequence[Rect]:
    '''
    Calculates random, non-overlapping positions for rectangles in given area
    :param rects: (List[Rect]) List of rectangles for which positions will be returned
    :param bounds: (Rect) Bounds of an area for positions calculation
    :return: (Sequence[Rect]) Rectangles at new positions
    '''
    bounding_rect = get_bounding_rect(rects)

    tree = BoundTree(bounds)
    tree.random_deep_divide(bounding_rect)

    areas = random.sample(list(tree), len(rects))
    for rect in rects:
        area = areas.pop()
        x = area.min.x + random.randint(0, area.width - rect.width)
        y = area.min.y + random.randint(0, area.height - rect.height)
        yield Rect(x, y, x + rect.width, y + rect.height)


def extract_rect(dataset: np.ndarray, rect: Rect) -> np.ndarray:
    '''
    Extracts rectangle from given dataset
    This function modifies the dataset
    :param dataset: (np.ndarray) Dataset with [x, y, z] format
    :param rect: (Rect) Area to extract
    :return: (np.ndarray) Extracted part of the dataset
    '''
    extracted = dataset[rect.min.x:rect.max.x, rect.min.y:rect.max.y, :].copy()
    dataset[rect.min.x:rect.max.x, rect.min.y:rect.max.y, :] = 0

    return extracted
