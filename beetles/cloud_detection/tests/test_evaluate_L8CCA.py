"""
Tests for L8CCA evaluation functions.

If you plan on using this implementation, please cite our work:
@INPROCEEDINGS{Grabowski2021IGARSS,
author={Grabowski, Bartosz and Ziaja, Maciej and Kawulok, Michal
and Nalepa, Jakub},
booktitle={IGARSS 2021 - 2021 IEEE International Geoscience
and Remote Sensing Symposium},
title={Towards Robust Cloud Detection in
Satellite Images Using U-Nets},
year={2021},
note={in press}}
"""

import numpy as np
from pathlib import Path

from cloud_detection.evaluate_L8CCA import build_rgb_scene_img


PATH_L8CCA = Path("datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial")
IMG_PATH = PATH_L8CCA / "Barren" / "LC81640502013179LGN01"
IMG_ID = "LC81640502013179LGN01"


class TestBuildRGBSceneImg:
    def test_build_rgb_scene_img(self):
        img = build_rgb_scene_img(
            path=IMG_PATH,
            img_id=IMG_ID
            )
        assert img.shape == (7741, 7551, 3)
        assert np.isclose(np.min(img), 0)
        assert np.isclose(np.max(img), 1)
