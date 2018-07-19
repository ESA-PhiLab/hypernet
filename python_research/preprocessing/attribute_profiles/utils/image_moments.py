def spatial_moments(array, j, i):
    moment = 0.0
    for pixel in array:
        moment += float(pixel.gray_level) * pow(pixel.x, j) * pow(pixel.y, i)
    return moment


def mass_center(array):
    moment10 = spatial_moments(array, 1, 0)
    moment00 = spatial_moments(array, 0, 0)
    moment01 = spatial_moments(array, 0, 1)
    x_mass_center = moment10 / moment00
    y_mass_center = moment01 / moment00
    return x_mass_center, y_mass_center


def central_moments(array, x_mass_center, y_mass_center):
    moment20 = 0
    moment02 = 0
    moment11 = 0
    for pixel in array:
        moment20 += float(pixel.gray_level) * pow(pixel.x - x_mass_center, 2)
        moment02 += float(pixel.gray_level) * pow(pixel.y - y_mass_center, 2)
        moment11 += float(pixel.gray_level) * (pixel.x - x_mass_center) * \
                    (pixel.y - y_mass_center)
    return moment20, moment02, moment11


def normalize_central_moments(moment20, moment02, moment11, moment00):
    denominator = pow(float(moment00), 2)
    norm_moment20 = moment20 / denominator
    norm_moment02 = moment02 / denominator
    norm_moment11 = moment11 / denominator
    return norm_moment20, norm_moment02, norm_moment11
