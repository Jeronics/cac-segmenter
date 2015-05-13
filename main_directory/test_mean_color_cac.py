import unittest
from MeanColorCAC import MeanColorCAC
import os.path


class TestLoadDataset(unittest.TestCase):
    def test_load_dataset(self):



if __name__ == '__main__':
    unittest.main()
    color = MeanColorCAC()
    color.train_model()
    print color.other
