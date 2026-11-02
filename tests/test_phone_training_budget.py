import unittest

from training.train_phone_classifier import limit_training_images


class Dataset:
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.targets[index]


class PhoneTrainingBudgetTests(unittest.TestCase):
    def test_selects_balanced_reproducible_subset(self):
        dataset = Dataset([0] * 7 + [1] * 4)

        first_subset, first_counts = limit_training_images(
            dataset,
            ["NOT_PHONE", "PHONE"],
            max_per_class=3,
            seed=42,
        )
        second_subset, second_counts = limit_training_images(
            dataset,
            ["NOT_PHONE", "PHONE"],
            max_per_class=3,
            seed=42,
        )

        self.assertEqual(len(first_subset), 6)
        self.assertEqual(first_counts, {"NOT_PHONE": 3, "PHONE": 3})
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(first_subset.indices, second_subset.indices)

    def test_zero_uses_all_training_images(self):
        dataset = Dataset([0] * 5 + [1] * 8)

        subset, counts = limit_training_images(
            dataset,
            ["NOT_PHONE", "PHONE"],
            max_per_class=0,
            seed=42,
        )

        self.assertIs(subset, dataset)
        self.assertEqual(len(subset), 13)
        self.assertEqual(counts, {"NOT_PHONE": 5, "PHONE": 8})


if __name__ == "__main__":
    unittest.main()
