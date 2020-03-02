import pytest
import numpy as np

from ml_intuition.data import utils


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
        assert len(utils._get_set_indices(labels, size, stratified)) == result

    @pytest.mark.parametrize("labels, size, result", [
        (labels_2_class, 0.5, np.array([0, 1, 2, 7, 8, 9])),
        (labels_3_class, 0.5, np.array([0, 2, 4])),
        (labels_4_class_unbalanced, 2, np.array([0, 1, 4, 5, 9, 10, 15, 16]))
    ])
    def test_if_returns_correct_indices(self, labels, size, result):
        assert np.all(np.equal(utils._get_set_indices(labels, size), result))


class TestTrainValTestSplit:
    data = np.ones((30, 1))
    labels = np.concatenate([np.zeros(10), np.ones(10), np.repeat(2, 10)])

    @pytest.mark.parametrize("data, labels, train_size, result", [
        (data, labels, 0.5, (10, 0, 10)),
        (np.ones((40, 1)), np.concatenate(
            [np.ones(5), np.repeat(2, 15), np.repeat(3, 18), np.repeat(4, 2)]),
         0.8, (29, 2, 9))
    ])
    def test_if_sets_have_correct_length(self, data, labels, train_size,
                                         result):
        train_x, train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(
            data, labels, train_size)
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
        train_x, train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(
            data, labels, train_size)
        assert len(train_x) == len(train_y) and len(val_x) == len(
            val_y) and len(test_x) == len(test_y)

    def test_if_sets_overlap(self, ):
        data = np.arange(30)
        train_x, train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(
            data, self.labels, 0.5, stratified=False)
        assert not np.any(np.equal(train_x, test_x))


class TestReshapeTo1DSamples:

    @pytest.mark.parametrize(
        "input_shape, output_shape, labels_shape, channels_idx", [
            ((10, 10, 3), (100, 3, 1), (10, 10), 2),
            ((3, 5, 5), (25, 3, 1), (5, 5), 0),
            ((5, 3, 1), (15, 1, 1), (5, 3), 2)
        ])
    def test_if_reshapes_correctly(self, input_shape, output_shape,
                                   labels_shape, channels_idx):
        data = np.zeros(input_shape)
        labels = np.zeros(labels_shape)
        reshaped_data, _ = utils.reshape_to_2d_samples(data, labels,
                                                       channels_idx)
        assert np.all(np.equal(reshaped_data.shape, output_shape))

    @pytest.mark.parametrize("data, channels_idx", [
        (np.arange(25).reshape((5, 5, 1)), 2),
        (np.arange(25).reshape((1, 5, 5)), 0)
    ])
    def test_if_data_matches_labels_after_reshape(self, data, channels_idx):
        labels = np.arange(25).reshape((5, 5))
        reshaped_data, reshaped_labels = utils.reshape_to_2d_samples(data,
                                                                     labels,
                                                                     channels_idx=channels_idx)
        assert np.all(np.equal(reshaped_data[:, 0, 0], reshaped_labels))
