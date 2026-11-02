import unittest

import numpy as np

from training.collect_phone_candidates import (
    is_near_duplicate,
    perceptual_hash,
)


class PhoneCollectorTests(unittest.TestCase):
    def test_identical_crops_have_identical_hashes(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[20:60, 35:85] = (40, 120, 220)

        self.assertEqual(perceptual_hash(image), perceptual_hash(image.copy()))

    def test_near_duplicate_filter_skips_matching_hash(self):
        image_hash = 0b101010

        self.assertTrue(is_near_duplicate(image_hash, [0b101011], 2))
        self.assertFalse(is_near_duplicate(image_hash, [0b000000], 2))

    def test_zero_distance_disables_filter(self):
        self.assertFalse(is_near_duplicate(1, [1], 0))


if __name__ == "__main__":
    unittest.main()
