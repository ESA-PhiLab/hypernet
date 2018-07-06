import numpy as np
from .max_tree.max_tree import MaxTree
from .utils.aux_functions import calculate_pca, normalize_pca, thinning_std, \
    thickening_std, thinning_area, thickening_area, invert_array


def generate_eap(x, areas, stds, n_components=0):
    pca = calculate_pca(x, n_components)
    x_max = float(np.amax(x))
    x_min = float(np.amin(x))
    pca = normalize_pca(pca, x_min, x_max)
    area_eap = []
    std_eap = []
    for index in range(0, n_components):
        pc = pca[:, :, index]
        tree_thickening = MaxTree(pc)
        tree_thinning = MaxTree(invert_array(pc))
        area_thickened = thickening_area(pc, tree_thickening, areas)
        area_thinned = thinning_area(pc, tree_thinning, areas)
        std_thickened = thickening_std(pc, tree_thickening, stds)
        std_thinned = thinning_std(pc, tree_thinning, stds)
        for i in range(0, area_thickened.shape[-1]):
            area_eap.append(area_thickened[:, :, i])
        area_eap.append(pc)
        for i in range(0, area_thinned.shape[-1]):
            area_eap.append(area_thinned[:, :, i])
        for i in range(0, std_thickened.shape[-1]):
            std_eap.append(std_thickened[:, :, i])
        std_eap.append(pc)
        for i in range(0, std_thinned.shape[-1]):
            std_eap.append(std_thinned[:, :, i])
    return area_eap, std_eap
