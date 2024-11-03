import unittest
import numpy as np
from cifar import load_batch, load_cifar10_data, split_into_nclasses, DATASET_PATH , split_into_classes, CLASSES

class TestCifar(unittest.TestCase):

    def test_load_batch(self):
        data, labels = load_batch(f"{DATASET_PATH}/data_batch_1")
        self.assertEqual(data.shape, (10000, 3, 32, 32))
        self.assertEqual(len(labels), 10000)

    def test_load_cifar10_data(self):
        train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
        self.assertEqual(train_data.shape, (50000, 3, 32, 32))
        self.assertEqual(len(train_labels), 50000)
        self.assertEqual(test_data.shape, (10000, 3, 32, 32))
        self.assertEqual(len(test_labels), 10000)

    def test_split_into_nclasses(self):
        train_data, train_labels, _, _ = load_cifar10_data(DATASET_PATH)
        num_classes = 5
        subset_train, subset_labels = split_into_nclasses(train_data, train_labels, num_classes)
        unique_labels = np.unique(subset_labels)
        self.assertEqual(len(unique_labels), num_classes)
        for label in unique_labels:
            self.assertIn(label, range(10))  # CIFAR-10 has labels from 0 to 9

    def test_split_into_classes(self):
        train_data, train_labels, _ , _ = load_cifar10_data(DATASET_PATH)
        selected_classes = ['airplane', 'cat' ,'pointyhats']  # Example classes
        
        subset_train, subset_labels = split_into_classes(train_data, train_labels, selected_classes)
        unique_labels = np.unique(subset_labels)
        print(selected_classes)
        print(unique_labels)
        # Check that all unique labels in the subset are within selected_classes
        for label in unique_labels:
            self.assertIn(CLASSES[label], selected_classes)


if __name__ == '__main__':
    unittest.main()