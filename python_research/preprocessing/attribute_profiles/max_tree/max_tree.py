import numpy as np
from copy import copy
from typing import List, Dict
from python_research.preprocessing.attribute_profiles.utils.data_types import Pixel
from python_research.preprocessing.attribute_profiles\
    .max_tree.attribute_matrix_construction import construct_matrix
from operator import attrgetter


IMAGE_DIMS = 2
IMPLEMENTED_ATTRIBUTES = ['area', 'stddev', 'diagonal', 'moment']
NOT_PROCESSED = -1


class MaxTree:
    def __init__(
        self, image: np.ndarray,
        attributes_to_compute: List[str] = None
    ):
        if attributes_to_compute is None:
            attributes_to_compute = ['area']
        for attribute in attributes_to_compute:
            if attribute not in IMPLEMENTED_ATTRIBUTES:
                raise ValueError(
                    "Attribute {} is not implemented".format(attribute)
                )
        if image.ndim != IMAGE_DIMS:
            raise ValueError(
                "Image should have {} dimensions, not {}".format(IMAGE_DIMS, image.ndim)
            )
        self.image = image
        self.parent = np.full(image.shape, NOT_PROCESSED, dtype=Pixel)
        self.zpar = np.zeros(image.shape, dtype=Pixel)
        self.s = []
        self.node_index = np.full(image.shape, -1)
        self.attribute_names = attributes_to_compute
        self._build_tree()
        self._compute_nani()

    def _get_neighbours(self, pixel: Pixel):
        neighbours = []
        adjacency = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == j == 0)]
        for dx, dy in adjacency:
            if 0 <= pixel.x + dx < self.image.shape[1] and 0 <= pixel.y + dy < self.image.shape[0]:
                x = pixel.x + dx
                y = pixel.y + dy
                neighbours.append(Pixel(x, y, self.image[y, x]))
        return neighbours

    def _sort_pixels(self):
        image_width = self.image.shape[1]
        for index, pixel_value in enumerate(self.image.flatten()):
            x = index % image_width
            y = int(index / image_width)
            self.s.append(Pixel(x, y, pixel_value))

        self.s = sorted(self.s, key=attrgetter('value'))

    def _find_root(self, pixel: Pixel):
        parent_pixel = self.zpar[pixel.coords]
        if parent_pixel != pixel:
            self.zpar[pixel.coords] = self._find_root(parent_pixel)
        return self.zpar[pixel.coords]

    def _canonize(self):
        for pixel in self.s:
            q = self.parent[pixel.coords]
            q_parent = self.parent[q.coords]
            if self.image[q.coords] == self.image[q_parent.coords]:
                self.parent[pixel.coords] = q_parent

    def _build_tree(self):
        self._sort_pixels()
        for pixel in reversed(self.s):
            self.parent[pixel.coords] = pixel
            self.zpar[pixel.coords] = pixel
            for neighbour in self._get_neighbours(pixel):
                if self.parent[neighbour.coords] != NOT_PROCESSED:
                    root = self._find_root(neighbour)
                    if root != pixel:
                        self.zpar[root.coords] = pixel
                        self.parent[root.coords] = pixel
        self._canonize()

    def _construct_attribute_matrices(self):
        attribute_matrices = dict()
        for attribute in self.attribute_names:
            attribute_matrices[attribute] = construct_matrix(attribute, self.image)
        return attribute_matrices

    def _compute_attributes(self):
        sorted_lvroots = []
        nlvroots = 0
        attribute_matrices = self._construct_attribute_matrices()
        for pixel in reversed(self.s):
            for attribute_matrix in attribute_matrices.values():
                attribute_matrix[self.parent[pixel.coords].coords] += attribute_matrix[pixel.coords]
            if (
                self.image[self.parent[pixel.coords].coords] != self.image[pixel.coords] or
                self.parent[pixel.coords] == pixel
            ):
                sorted_lvroots.append(pixel)
                nlvroots += 1
        return sorted_lvroots, nlvroots, attribute_matrices

    def _fill_nani(
        self,
        sorted_lvroots: List,
        attribute_matrices: Dict[str, np.ndarray]
    ):
        for index, pixel in enumerate(reversed(sorted_lvroots)):
            self.node_index[pixel.coords] = index
            self.parent_gray_level_relation[index, 0] = self.node_index[
                self.parent[pixel.coords].coords
            ]
            self.parent_gray_level_relation[index, 1] = self.image[pixel.coords]
            for attribute_name in self.attribute_values.keys():
                self.attribute_values[attribute_name][index] = attribute_matrices[
                    attribute_name
                ][pixel.coords].get()

    def _fill_remaining_positions(self):
        node_index_width = self.node_index.shape[1]
        for index, value in enumerate(self.node_index.flatten()):
            x = index % node_index_width
            y = int(index / node_index_width)
            if self.node_index[y, x] == -1:
                self.node_index[y, x] = self.node_index[
                    self.parent[y, x].coords
                ]

    def _compute_nani(self):
        sorted_lvroots, nlvroots, attribute_matrices = self._compute_attributes()
        self.parent_gray_level_relation = np.zeros((nlvroots, 2))
        self.attribute_values = {
            attribute_name: np.zeros((nlvroots,)) for attribute_name in self.attribute_names
        }
        self._fill_nani(sorted_lvroots, attribute_matrices)
        self._fill_remaining_positions()

    def _define_removed_nodes(self, attribute: str, threshold):
        to_keep = self.attribute_values[attribute] < threshold
        return ~to_keep

    def _direct_filter(self, attribute, threshold):
        to_keep = self._define_removed_nodes(attribute, threshold)
        to_keep[0] = True
        parent = copy(self.parent_gray_level_relation[:, 0])
        lut = self._update_parents(parent, to_keep)
        lut = self._update_lut(lut, to_keep)
        node_index = self._update_node_index(lut)
        return node_index

    def _update_parents(self, parent, to_keep):
        M = self.parent_gray_level_relation.shape[0]
        nearest_ancestor_kept = [0 for _ in range(0, M)]
        lut = [x for x in range(0, M)]
        for i in range(0, M):
            if not to_keep[i]:
                temp = nearest_ancestor_kept[int(parent[i])]
                nearest_ancestor_kept[i] = temp
                lut[i] = lut[temp]
            else:
                nearest_ancestor_kept[i] = i
                parent[i] = nearest_ancestor_kept[int(parent[i])]
        return lut

    def _update_lut(self, lut: List, to_keep: List[bool]):
        M = self.parent_gray_level_relation.shape[0]
        index_fix = [None for _ in range(0, M)]
        index_fix[0] = ~to_keep[0]
        for i in range(1, M):
            index_fix[i] = index_fix[i - 1] + ~to_keep[i]
            lut[i] = lut[i] + index_fix[i]
        return lut

    def _update_node_index(self, lut: List):
        image_width = self.node_index.shape[1]
        node_index = copy(self.node_index)
        for index, pixel in enumerate(node_index.flatten()):
            x = index % image_width
            y = int(index / image_width)
            node_index[y, x] = lut[node_index[y, x]]
        return node_index

    def _reconstitute_image(self, node_index):
        image = copy(self.image)
        image_width = image.shape[1]
        for index, pixel in enumerate(image.flatten()):
            x = index % image_width
            y = int(index / image_width)
            image[y, x] = self.parent_gray_level_relation[[node_index[y, x]], 1]
        return image

    def filter(self, attribute: str, threshold):
        node_index = self._direct_filter(attribute, threshold)
        return self._reconstitute_image(node_index)
