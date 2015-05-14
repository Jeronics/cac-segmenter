import unittest
from MeanColorCAC import MeanColorCAC
import numpy as np
import pandas as pd
import os.path


class TestLoadDataset(unittest.TestCase):
    def test_load_dataset(self):
        color_cac = MeanColorCAC()


class TestPartitionDataset(unittest.TestCase):
    def test_partition_dataset_with_first_test_dataset_of_30_instances_and_CV_5(self):
        '''
        30
            [Test 6] Train 6  Train 6  Train 6  Train 6
        '''
        dataset = pd.DataFrame(np.ones([30, 4]))
        color_cac = MeanColorCAC()
        expected_test = pd.DataFrame(np.ones([6, 4]))
        _, predicted_test = color_cac._partition_dataset(dataset, 0, 5)
        self.assertEqual(expected_test.shape, predicted_test.shape)

    def test_partition_dataset_with_first_train_dataset_of_30_instances_and_CV_5(self):
        '''
        30
            Test 6 [ Train 6  Train 6  Train 6  Train 6 ]
        '''
        dataset = pd.DataFrame(np.ones([30, 4]))
        color_cac = MeanColorCAC()
        expected_train = pd.DataFrame(np.ones([24, 4]))
        predicted_train, _ = color_cac._partition_dataset(dataset, 0, 5)
        self.assertEqual(expected_train.shape, predicted_train.shape)


    def test_partition_dataset_with_last_train_dataset_of_30_instances_and_CV_5(self):
        '''
        30
            [ Train 6  Train 6  Train 6  Train 6 ] Test 6
        '''
        dataset = pd.DataFrame(np.ones([30, 4]))
        color_cac = MeanColorCAC()
        expected_train = pd.DataFrame(np.ones([25, 4]))
        predicted_train, _ = color_cac._partition_dataset(dataset, 4, 5)
        self.assertEqual(expected_train.shape, predicted_train.shape)

    def test_partition_dataset_with_last_test_dataset_of_30_instances_and_CV_5(self):
        '''
        30
            Train 6  Train 6  Train 6  Train 6  [ Test 6 ]
        '''
        dataset = pd.DataFrame(np.ones([30, 4]))
        color_cac = MeanColorCAC()
        expected_test = pd.DataFrame(np.ones([5, 4]))
        _, predicted_test = color_cac._partition_dataset(dataset, 4, 5)
        self.assertEqual(expected_test.shape, predicted_test.shape)

    def test_partition_dataset_with_first_test_dataset_of_31_instances_and_CV_5(self):
        '''
        31
            [Test 6]  Train 6  Train 6  Train 6  Train 6
        '''
        dataset = pd.DataFrame(np.ones([31, 4]))
        color_cac = MeanColorCAC()
        expected_test = pd.DataFrame(np.ones([6, 4]))
        _, predicted_test = color_cac._partition_dataset(dataset, 0, 5)
        self.assertEqual(expected_test.shape, predicted_test.shape)



    def test_partition_dataset_with_first_train_dataset_of_31_instances_and_CV_5(self):
        '''
        31
            Test 6 [Train 6  Train 6  Train 6  Train 6]
        '''
        dataset = pd.DataFrame(np.ones([31, 4]))
        color_cac = MeanColorCAC()
        expected_train = pd.DataFrame(np.ones([25, 4]))
        predicted_train, _ = color_cac._partition_dataset(dataset, 0, 5)
        self.assertEqual(expected_train.shape, predicted_train.shape)


    def test_partition_dataset_with_last_test_dataset_of_31_instances_and_CV_5(self):
        '''
        31
            Train 6  Train 6  Train 6  Train 6 [ Test 6 ]
        '''
        dataset = pd.DataFrame(np.ones([31, 4]))
        color_cac = MeanColorCAC()
        expected_test = pd.DataFrame(np.ones([6, 4]))
        _, predicted_test = color_cac._partition_dataset(dataset, 4, 5)
        self.assertEqual(expected_test.shape, predicted_test.shape)


    def test_partition_dataset_with_last_train_dataset_of_31_instances_and_CV_5(self):
        '''
        31
            Train 6  Train 6  Train 6  Train 6 [ Test 6 ]
        '''
        dataset = pd.DataFrame(np.ones([31, 4]))
        color_cac = MeanColorCAC()
        expected_train = pd.DataFrame(np.ones([25, 4]))
        predicted_train, _ = color_cac._partition_dataset(dataset, 4, 5)
        self.assertEqual(expected_train.shape, predicted_train.shape)


if __name__ == '__main__':
    unittest.main()
