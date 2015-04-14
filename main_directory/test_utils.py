__author__ = 'jeroni'
import unittest
import utils
import numpy as np


class MyTest(unittest.TestCase):
    def setUp(self):
        coordinates1 = np.array([[1, 1], [1, 1], [0, 0]])
        size1 = [3, 3]
        b1 = False

        coordinates2 = np.array([[-0.4, 205], [2, 0]])
        size2 = [500, 200]
        b2 = False

        coordinates3 = np.array([[-0.4, 205], [200, -2]])
        size3 = [500, 200]
        b3 = True

        coordinates4 = np.array([[0, 200]])
        size4 = [500, 200]
        b4 = True

        coordinates5 = np.array([[400, 190]])
        size5 = [500, 200]
        b5 = False

        self.expected_values = (
            ((coordinates1, size1), b1),
            ((coordinates2, size2), b2),
            ((coordinates3, size3), b3),
            ((coordinates4, size4), b4),
            ((coordinates5, size5), b5),
        )

    def test_coordinates(self):
        for input_vars, expected in self.expected_values:
            predicted = utils.cage_out_of_the_picture(*input_vars)
            self.assertEqual(predicted, expected),


if __name__ == '__main__':
    unittest.main()