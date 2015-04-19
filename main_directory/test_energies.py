__author__ = 'jeroni'
import unittest
import energies
import numpy as np


class TestVertexConstraint(unittest.TestCase):
    '''

    '''

    def setUp(self):
        '''

        :return:
        '''
        '''
        case 1: Only two points influence each other

             (0,1)*
             (0,0)*          * (10,0)

        '''
        vertices1 = np.array([
            [0, 1],
            [10, 0],
            [0, 0]
        ])
        d1 = 10
        gradients1 = np.array([
            [0, 18],
            [0, 0],
            [0, -18]
        ])
        energy1 = 81

        '''
        case 2: Triangle with 45 a deg. angle at (0,0)

                           * (2*sqrt(2),2*sqrt(2))

                (0,0) *        * (4*sqrt(2),0)

        '''
        vertices2 = np.array([
            [0, 0],
            [2 * np.sqrt(2), 2 * np.sqrt(2)],
            [4 * np.sqrt(2), 0]
        ])
        d2 = 10
        gradients2 = np.array([
            [-20 + 2 * np.sqrt(2), -6 * np.sqrt(2)],
            [0, 12 * np.sqrt(2)],
            [20 - 2 * np.sqrt(2), -6 * np.sqrt(2)]
        ])
        energy2 = 36 * 2 + np.power(10 - 4 * np.sqrt(2), 2)

        '''
            Case 3: Equilateral triangle

                     * (2,2*sqrt(3))

            (0,0) *     * (4,0)

        '''
        vertices3 = np.array([
            [0, 0],
            [2, 2 * np.sqrt(3)],
            [4, 0]
        ])
        d3 = 10
        gradients3 = np.array([
            [-18, -6 * np.sqrt(3)],
            [0, 12 * np.sqrt(3)],
            [18, -6 * np.sqrt(3)]
        ])
        energy3 = 36 * 3

        '''

            Case 4: Case 1 with inverse order of points.

        '''
        vertices4 = np.array([
            [0, 0],
            [10, 0],
            [0, 1]
        ])
        d4 = 10
        gradients4 = np.array([
            [0, -18],
            [0, 0],
            [0, 18],
        ])
        energy4 = 81

        self.expected_values = (
            ((vertices1, d1), gradients1, energy1),
            ((vertices1 + 1212, d1), gradients1, energy1),
            ((vertices1 - 233, d1), gradients1, energy1),
            ((vertices2, d2), gradients2, energy2),
            ((vertices3, d3), gradients3, energy3),
            ((vertices4, d4), gradients4, energy4),

        )

    def test_vertex_constraint_grad(self):
        '''
        Tests whether the vertex_constraint_grad function works.
        '''
        for input_vars, expected, _ in self.expected_values:
            predicted = energies.grad_vertex_constraint(*input_vars)
            self.assertEqual(np.linalg.norm(predicted - expected) < 0.0001, True)

    def test_vertex_constraint(self):
        '''
        Tests whether the vertex_constraint function works.
        '''
        for input_vars, _, expected in self.expected_values:
            predicted = energies.vertex_constraint(*input_vars)
            self.assertEqual(abs(predicted - expected) < 0.0001, True)


class TestEdgeConstraint(unittest.TestCase):
    '''

    '''

    def setUp(self):
        '''

        :return:
        '''
        '''
        case 1: A point is influenced by an edge.

                  (1,5) *
             (0,0)*-----------* (10,0)

        '''
        vertices1 = np.array([
            [0, 0],
            [5, 1],
            [10, 0]
        ])
        d1 = 5
        gradients1 = np.array([
            [0, 0],
            [0, 8],
            [0, 0]
        ])
        energy1 = 16

        '''
        case 2: Case 1 in inverse order
        '''
        vertices2 = np.array([
            [0, 0],
            [10, 0],
            [5, 1]
        ])
        d2 = 5
        gradients2 = np.array([
            [0, 0],
            [0, 0],
            [0, 8]
        ])
        energy2 = 16

        '''
        case 3: A rectangle

            (0,3) * - - * (1,3)
                  |     |
                  |     |
                  |     |
            (0,0) * - - * (1,0)

        '''
        vertices3 = np.array([
            [0, 0],
            [0, 3],
            [1, 3],
            [1, 0]
        ])
        d3 = 5
        gradients3 = np.array([
            [-8, -4],
            [-8, 4],
            [8, 4],
            [8, -4]
        ])
        energy3 = 4 * 2 * 2 + 4 * 4 * 4

        self.expected_values = (
            ((vertices1, d1), gradients1, energy1),
            ((vertices1 + 1212, d1), gradients1, energy1),
            ((vertices1 - 322, d1), gradients1, energy1),
            ((vertices2, d2), gradients2, energy2),
            ((vertices3, d3), gradients3, energy3),
            # ((vertices4, d4), gradients4, energy4),

        )

    def test_edge_constraint_grad(self):
        '''
        Tests whether the vertex_constraint_grad function works.
        '''
        for input_vars, expected, _ in self.expected_values:
            predicted = energies.grad_edge_constraint(*input_vars)
            self.assertEqual(np.linalg.norm(predicted - expected) < 0.0001, True)

    def test_vertex_constraint(self):
        '''
        Tests whether the edge_constraint function works.
        '''
        for input_vars, _, expected in self.expected_values:
            predicted = energies.edge_constraint(*input_vars)
            self.assertEqual(abs(predicted - expected) < 0.0001, True)


class TestColorMean(unittest.TestCase):
    '''

    '''

    def setUp(self):
        '''

        :return:
        '''
        '''
        case 1: Red
        '''
        import numpy as np
        import utils

        color1 = np.ones([15, 20, 3])


        # Red
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 0.
        col[:, :, 2] *= 0.
        red_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_red = utils.ImageClass(col)
        h_red = 0
        s_red = 100
        i_red = 100

        # Green
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 0.
        green_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_green = utils.ImageClass(col)
        h_green = 2*np.pi/3.
        s_green = 100
        i_green = 100

        # Blue
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 0.
        col[:, :, 2] *= 255.
        blue_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_blue = utils.ImageClass(col)
        h_blue = 3*np.pi/2.
        s_blue = 100
        i_blue = 100

        # Yellow
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 0.
        yellow_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_yellow = utils.ImageClass(col)
        h_yellow = np.pi/3.
        s_yellow = 100
        i_yellow = 100

        # Pink
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 0.
        col[:, :, 2] *= 255.
        pink_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_pink = utils.ImageClass(col)
        h_pink = 7*np.pi/3.
        s_pink = 100
        i_pink = 100

        # Clear Blue
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 255.
        clear_blue_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_clear_blue = utils.ImageClass(col)
        h_clear_blue = np.pi
        s_clear_blue = 100
        i_clear_blue = 100

        # Black
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 0.
        col[:, :, 2] *= 0.
        black_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_black = utils.ImageClass(col)
        h_black = 0
        s_black = 0
        i_black = 0

        # White
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 255.
        white_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_white = utils.ImageClass(col)
        h_white = 0
        s_white = 0
        i_white = 100

        self.expected_values = (
            ((red_coordinates, im_red.image), h_red, s_red, i_red),
            ((green_coordinates, im_green.image), h_green, s_green, i_green),
            ((blue_coordinates, im_blue.image), h_blue, s_blue,i_blue),
            ((yellow_coordinates, im_yellow.image), h_yellow, s_yellow, i_yellow),
            ((pink_coordinates, im_pink.image), h_pink, s_pink, i_pink),
            ((clear_blue_coordinates, im_clear_blue.image), h_clear_blue, s_clear_blue, i_clear_blue),
            ((black_coordinates, im_black.image), h_black, s_black, i_black),
            ((white_coordinates, im_white.image), h_white, s_white, i_white),
        )

    def test_rgb_to_green(self):
        '''
        Tests whether the vertex_constraint_grad function works.
        '''
        for input_vars, h, s, i in self.expected_values:
            pred_h,pred_s, pred_i = energies.rgb_to_hsi(*input_vars)
            # self.assertEqual(np.linalg.norm(pred_h - h) < 0.0001, True)
            # self.assertEqual(np.linalg.norm(pred_s - s) < 0.0001, True)
            self.assertEqual(np.linalg.norm(pred_i - i) < 0.0001, True)

    def test_vertex_constraint(self):
        '''
        Tests whether the edge_constraint function works.
        '''
        for input_vars, _, expected in self.expected_values:
            predicted = energies.edge_constraint(*input_vars)
            self.assertEqual(abs(predicted - expected) < 0.0001, True)


if __name__ == '__main__':
    unittest.main()