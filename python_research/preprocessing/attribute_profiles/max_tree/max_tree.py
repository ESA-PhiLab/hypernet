import numpy as np
from copy import copy
from ..utils.data_types import Pixel, StdDevIncrementally
from ..utils.aux_functions import radix_sort, construct_std_dev_matrix, \
    push_root


class MaxTree:
    def __init__(self, image):
        self.image = image
        self.parent = np.full(image.shape, -1, dtype=Pixel)
        self.zpar = np.zeros(image.shape, dtype=Pixel)
        self.s = []
        self.node_index = np.full(image.shape, -1)
        self.node_array = []
        self._build_tree()
        self._compute_nani()

    def _get_neighbours(self, pixel: Pixel):
        neighbours = []
        adjacency = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if
                     not (i == j == 0)]
        for dx, dy in adjacency:
            if 0 <= pixel.x + dx < self.image.shape[1] and 0 <= pixel.y + dy \
                    < self.image.shape[0]:
                x = pixel.x + dx
                y = pixel.y + dy
                gray_level = self.image[y, x]
                neighbours.append(Pixel(x, y, gray_level))
        return neighbours

    def _sort_pixels(self):
        image_width = self.image.shape[1]
        for index, pixel_value in enumerate(self.image.flatten()):
            x = index % image_width
            y = int(index / image_width)
            self.s.append(Pixel(x, y, pixel_value))
        self.s = radix_sort(self.s)

    def _find_root(self, pixel):
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
                if self.parent[neighbour.coords] != -1:
                    root = self._find_root(neighbour)
                    if root != pixel:
                        self.zpar[root.coords] = pixel
                        self.parent[root.coords] = pixel
        self._canonize()

    def _compute_nani(self):
        sorted_lvroots = []
        nlvroots = 0
        area = np.ones(self.image.shape)
        standard_deviation = construct_std_dev_matrix(self.image)
        for pixel in reversed(self.s):
            area[self.parent[pixel.coords].coords] += area[pixel.coords]
            std_dev = standard_deviation[self.parent[pixel.coords].coords] + \
                      standard_deviation[pixel.coords]
            standard_deviation[self.parent[pixel.coords].coords] = std_dev
            if self.image[self.parent[pixel.coords].coords] != self.image[
                pixel.coords] or self.parent[pixel.coords] == pixel:
                sorted_lvroots = push_root(pixel, sorted_lvroots)
                nlvroots += 1
        k = 0
        self.node_array = np.zeros((nlvroots, 4), dtype=np.uint64)
        width = standard_deviation.shape[1]
        for index, std_dev in enumerate(standard_deviation.flatten()):
            if std_dev.n_samples == 1:
                x = index % width
                y = int(index / width)
                standard_deviation[y, x] = StdDevIncrementally(mean=0,
                                                               n_samples=1,
                                                               variance=0)

        for pixel in reversed(sorted_lvroots):
            self.node_index[pixel.coords] = k
            self.node_array[k, 0] = self.node_index[self.parent[
                pixel.coords].coords]
            self.node_array[k, 1] = self.image[pixel.coords]
            self.node_array[k, 2] = area[pixel.coords]
            self.node_array[k, 3] = standard_deviation[pixel.coords].get_std()
            k += 1
        for i, row in enumerate(self.node_index):
            for j, pixel in enumerate(row):
                if self.node_index[i, j] == -1:
                    self.node_index[i, j] = self.node_index[self.parent[i,
                                                                        j].coords]
        self.node_array[0, 2] = self.node_array[0, 2] / 2

    def _define_removed_nodes(self, attribute, threshold):
        if attribute == 'area':
            i = 2
        elif attribute == 'stdev':
            i = 3
        else:
            raise ValueError(
                "The attribute {} is not implemented.".format(attribute))
        to_keep = self.node_array[:, i] < threshold
        return ~to_keep

    def _direct_filter(self, attribute, threshold):
        to_keep = self._define_removed_nodes(attribute, threshold)
        to_keep[0] = True
        M = self.node_array.shape[0]
        parent = copy(self.node_array[:, 0])
        nearest_ancestor_kept = [0 for _ in range(0, M)]
        lut = [x for x in range(0, M)]
        index_fix = [None for _ in range(0, M)]
        for i in range(0, M):
            if not to_keep[i]:
                temp = nearest_ancestor_kept[parent[i]]
                nearest_ancestor_kept[i] = temp
                lut[i] = lut[temp]
            else:
                nearest_ancestor_kept[i] = i
                parent[i] = nearest_ancestor_kept[parent[i]]
        index_fix[0] = ~to_keep[0]
        for i in range(1, M):
            index_fix[i] = index_fix[i - 1] + ~to_keep[i]
            lut[i] = lut[i] + index_fix[i]
        for i in range(0, M):
            parent[i] = lut[parent[i]]
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
            image[y, x] = self.node_array[[node_index[y, x]], 1]
        return image

    def filter(self, attribute, threshold):
        node_index = self._direct_filter(attribute, threshold)
        return self._reconstitute_image(node_index)
