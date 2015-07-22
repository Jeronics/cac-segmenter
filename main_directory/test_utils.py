__author__ = 'jeroni'
import unittest
import utils
import numpy as np


class TestCoordinates(unittest.TestCase):
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
            self.assertEqual(predicted, expected)


class TestImageClass(unittest.TestCase):
    def test_hsi_image(self):
        filename = 'images_tester/test_hsi.png'
        '''
        pi     0     pi/2    pi
       2pi/3  pi/3  3pi/3    0
       3pi/2  5pi/3  4pi/3 4pi/3
        '''

        color1 = np.zeros([4, 3, 3])
        # Image
        col = color1.copy()
        col[0, 0] = [0., 255., 255.]
        col[1, 0] = [255., 0., 0.]
        col[2, 0] = [127.5, 255., 0.]
        col[3, 0] = [0., 255., 255.]
        col[0, 1] = [0., 255., 0.]
        col[1, 1] = [255., 255., 0.]
        col[2, 1] = [127.5, 0., 255.]
        col[3, 1] = [255., 0., 0.]
        col[0, 2] = [127.5, 0., 255.]
        col[1, 2] = [255., 0., 255.]
        col[2, 2] = [0., 0., 255.]
        col[3, 2] = [0., 0., 255.]

        pi_ = np.pi
        expected_hsi = np.zeros([4, 3, 3])
        expected_hsi[0, 0] = [pi_, 255., 255.]
        expected_hsi[1, 0] = [0, 0., 0.]
        expected_hsi[2, 0] = [pi_ / 2., 255., 0.]
        expected_hsi[3, 0] = [pi_, 255., 255.]
        expected_hsi[0, 1] = [2 * pi_ / 3, 255., 0.]
        expected_hsi[1, 1] = [pi_ / 3., 255., 0.]
        expected_hsi[2, 1] = [3 * pi_ / 2., 0., 255.]
        expected_hsi[3, 1] = [0., 0., 0.]
        expected_hsi[0, 2] = [3 * pi_ / 2., 0., 255.]
        expected_hsi[1, 2] = [5 * pi_ / 3., 0., 255.]
        expected_hsi[2, 2] = [4 * pi_ / 3., 0., 255.]
        expected_hsi[3, 2] = [4 * pi_ / 3., 0., 255.]
        im_color1 = utils.ImageClass(im=col)
        predicted = im_color1.hsi_image
        self.assertEqual(np.linalg.norm(predicted[:, :, 0] - expected_hsi[:, :, 0]) < 0.0001, True)


if __name__ == '__main__':
    unittest.main()