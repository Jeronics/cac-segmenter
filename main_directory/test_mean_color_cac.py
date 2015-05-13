import unittest
from MeanColorCAC import MeanColorCAC
import numpy as np
import pandas as pd
import os.path


class TestLoadDataset(unittest.TestCase):
    def test_load_dataset(self):
        color_cac = MeanColorCAC()

class TestCrossValidation(unittest.TestCase):
    def test_cross_valitation_with_dataset_of_30_instances_and_CV_5(self):
        dataset=pd.DataFrame(np.zeros([30,4]))
        color_cac = MeanColorCAC()
        color_cac._cross_validation(dataset,CV=5)
        expected=




if __name__ == '__main__':
    unittest.main()
    color = MeanColorCAC()
    color.train_model()
    print color.other
