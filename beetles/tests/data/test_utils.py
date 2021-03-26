import pytest
import numpy as np

from ml_intuition.data import utils, preprocessing


class TestShufflingArrays:

    def test_if_arrays_are_modified_in_place(self):
        array = np.arange(10)
        utils.shuffle_arrays_together([array])
        assert not np.all(np.equal(array, np.arange(10)))

    @pytest.mark.parametrize("arrays", [
        ([np.arange(4), np.arange(4)]),
        ([np.arange(10), np.arange(10), np.arange(10)]),
        ([np.arange(3), np.arange(3), np.arange(3), np.arange(3)])
    ])
    def test_if_shuffled_arrays_have_same_order(self, arrays):
        utils.shuffle_arrays_together(arrays)
        assert np.all([np.equal(x1, arrays[0]) for x1 in arrays[1:]])

    @pytest.mark.parametrize("arrays, seed", [
        ([np.arange(4), np.arange(4)], 123),
        ([np.arange(10), np.arange(10), np.arange(10)], 5),
        ([np.arange(3), np.arange(3), np.arange(3), np.arange(3)], 69)
    ])
    def test_if_works_for_different_seeds(self, arrays, seed):
        utils.shuffle_arrays_together(arrays, seed)
        assert np.all([np.equal(x, arrays[0]) for x in arrays[1:]])

    def test_if_throws_for_arrays_with_different_sizes(self):
        array1 = np.arange(10)
        array2 = np.arange(5)
        with pytest.raises(AssertionError):
            utils.shuffle_arrays_together([array1, array2])


class TestGetSetIndices:
    labels_2_class = np.concatenate([np.zeros(7), np.ones(7)])
    labels_3_class = np.array([1, 1, 2, 2, 3, 3])
    labels_4_class_unbalanced = np.concatenate([np.zeros(4), np.ones(5),
                                                np.repeat(2, 6),
                                                np.repeat(3, 7)])

    @pytest.mark.parametrize("labels, size, stratified, result", [
        (labels_2_class, 0.7, True, 8),
        (labels_2_class, 5, True, 10),
        (labels_2_class, 0.8, False, 11),
        (labels_2_class, 10, False, 10),
        (labels_3_class, 0.5, True, 3),
        (labels_3_class, 0.5, False, 3),
        (labels_3_class, 1, True, 3),
        (labels_3_class, 5, False, 5),
        (labels_4_class_unbalanced, 0.3, True, 5),
        (labels_4_class_unbalanced, 0.3, False, 6),
        (labels_4_class_unbalanced, 3, True, 12),
        (labels_4_class_unbalanced, 15, False, 15)
    ])
    def test_if_returns_correct_amount(self, labels, size, stratified, result):
        assert len(
            preprocessing._get_set_indices(size, labels, stratified)) == result

    @pytest.mark.parametrize("labels, size, result", [
        (labels_2_class, 0.5, np.array([0, 1, 2, 7, 8, 9])),
        (labels_3_class, 0.5, np.array([0, 2, 4])),
        (labels_4_class_unbalanced, 2, np.array([0, 1, 4, 5, 9, 10, 15, 16]))
    ])
    def test_if_returns_correct_indices(self, labels, size, result):
        assert np.all(np.equal(
            preprocessing._get_set_indices(size, labels), result))


class TestTrainValTestSplit:
    data = np.ones((30, 1))
    labels = np.concatenate([np.zeros(10), np.ones(10), np.repeat(2, 10)])

    @pytest.mark.parametrize("data, labels, train_size, result", [
        (data, labels, 0.5, (15, 0, 15)),
        (np.ones((40, 1)), np.concatenate(
            [np.ones(5), np.repeat(2, 15), np.repeat(3, 18), np.repeat(4, 2)]),
         0.8, (29, 2, 9))
    ])
    def test_if_sets_have_correct_length(self, data, labels, train_size,
                                         result):
        train_x, train_y, val_x, val_y, test_x, test_y = preprocessing. \
            train_val_test_split(data, labels, train_size)
        assert len(train_x) == result[0] and len(val_x) == result[1] and len(
            test_x) == result[2]

    @pytest.mark.parametrize("data, labels, train_size, result", [
        (data, labels, 0.5, (10, 0, 10)),
        (np.ones((40, 1)), np.concatenate(
            [np.ones(5), np.repeat(2, 15), np.repeat(3, 18), np.repeat(4, 2)]),
         0.8, (29, 2, 9))
    ])
    def test_if_x_and_y_have_same_length(self, data, labels, train_size,
                                         result):
        train_x, train_y, val_x, val_y, test_x, test_y = preprocessing. \
            train_val_test_split(data, labels, train_size)
        assert len(train_x) == len(train_y) and len(val_x) == len(
            val_y) and len(test_x) == len(test_y)

    def test_if_sets_overlap(self, ):
        data = np.arange(30)
        train_x, train_y, val_x, val_y, test_x, test_y = preprocessing. \
            train_val_test_split(data, self.labels, 0.5, stratified=False)
        assert len(np.intersect1d(train_x, test_x)) == 0
