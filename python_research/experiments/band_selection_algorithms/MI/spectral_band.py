class SpectralBand(object):
    def __init__(self, histogram, joint_histogram, band_index):
        pass
        self.histogram = histogram
        self.joint_histogram = joint_histogram
        self.band_index = band_index
        self.mutual_information = None

    def set_mutual_information(self, mutual_information):
        self.mutual_information = mutual_information
