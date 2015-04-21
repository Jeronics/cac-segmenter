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
        s_red = 255.
        i_red = 85
        mean_red = 0

        # Green
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 0.
        green_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_green = utils.ImageClass(col)
        h_green = 2 * np.pi / 3.
        s_green = np.sqrt(np.power(255 / 2., 2) + 3 * np.power(255 / 2., 2))
        i_green = 85
        mean_green = 2 * np.pi / 3.

        # Blue
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 0.
        col[:, :, 2] *= 255.
        blue_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_blue = utils.ImageClass(col)
        h_blue = 4 * np.pi / 3.
        s_blue = np.sqrt(np.power(255 / 2., 2) + 3 * np.power(255 / 2., 2))
        i_blue = 85
        mean_blue = 4 * np.pi / 3.

        # Yellow
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 0.
        yellow_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_yellow = utils.ImageClass(col)
        h_yellow = np.pi / 3.
        s_yellow = np.sqrt(np.power(255 - 255 / 2., 2) + 3 * np.power(255 / 2., 2))
        i_yellow = 170
        mean_yellow = np.pi / 3.

        # Pink
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 0.
        col[:, :, 2] *= 255.
        pink_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_pink = utils.ImageClass(col)
        h_pink = 5 * np.pi / 3.
        s_pink = np.sqrt(np.power(255 - 255 / 2., 2) + 3 * np.power(255 / 2., 2))
        i_pink = 170
        mean_pink = 5 * np.pi / 3.

        # Clear Blue
        col = color1.copy()
        col[:, :, 0] *= 0.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 255.
        clear_blue_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_clear_blue = utils.ImageClass(col)
        h_clear_blue = np.pi
        s_clear_blue = 255.
        i_clear_blue = 170
        mean_clear_blue = np.pi

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
        mean_black = 0

        # White
        col = color1.copy()
        col[:, :, 0] *= 255.
        col[:, :, 1] *= 255.
        col[:, :, 2] *= 255.
        white_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_white = utils.ImageClass(col)
        h_white = 0
        s_white = 0
        i_white = 255
        mean_white = 0

        # Red-Yellow-black-white
        col = color1.copy()

        col[:, :5, 0], col[:, :5, 1], col[:, :5, 2] = 255., 0., 0.
        col[:, 5:10, 0], col[:, 5:10, 1], col[:, 5:10, 2] = 255., 255., 0
        col[:, 10:15, 0], col[:, 10:15, 1], col[:, 10:15, 2] = 0., 0., 0.
        col[:, 15:20, 0], col[:, 15:20, 1], col[:, 15:20, 2] = 255., 255., 255.
        mix1_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_mix1 = utils.ImageClass(col)
        h_mix1 = None  # Should be a vector we do not use it for our case
        s_mix1 = None  # Idem.
        i_mix1 = None  # Idem.
        mean_mix1 = np.pi / 6.

        self.expected_values = (
            ((red_coordinates, im_red.image), h_red, s_red, i_red, mean_red),
            ((green_coordinates, im_green.image), h_green, s_green, i_green, mean_green),
            ((blue_coordinates, im_blue.image), h_blue, s_blue, i_blue, mean_blue),
            ((yellow_coordinates, im_yellow.image), h_yellow, s_yellow, i_yellow, mean_yellow),
            ((pink_coordinates, im_pink.image), h_pink, s_pink, i_pink, mean_pink),
            ((clear_blue_coordinates, im_clear_blue.image), h_clear_blue, s_clear_blue, i_clear_blue, mean_clear_blue),
            ((black_coordinates, im_black.image), h_black, s_black, i_black, mean_black),
            ((white_coordinates, im_white.image), h_white, s_white, i_white, mean_white),
        )

        self.color_pairs = (
            (
                np.array([
                    [0],
                    [0],
                    [0.01],
                    [0],
                    [2 * np.pi / 3.],
                ]),
                np.array([
                    [0],
                    [np.pi],
                    [np.pi + 0.01],
                    [3 * np.pi / 2.],
                    [3 * np.pi / 2.]
                ])
            ), np.array([
                [0],
                [np.pi],
                [np.pi],
                [np.pi / 2.],
                [5 * np.pi / 6.]
            ])
        )

        color2 = np.ones([4, 3, 3])

        col = color2.copy()
        col[:2, :, 0], col[:2, :, 1], col[:2, :, 2] = 255., 0., 0.
        col[2:, :, 0], col[2:, :, 1], col[2:, :, 2] = 0., 255., 0.
        mix2_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_mix2 = utils.ImageClass(col)
        mean_mix2 = np.pi / 3.
        energy_mix2_region = 4 * np.pi * np.pi / 3.

        col = color2.copy()
        col[:, :1, 0], col[:, :1, 1], col[:, :1, 2] = 255., 0., 0.
        col[:, 1:2, 0], col[:, 1:2, 1], col[:, 1:2, 2] = 0., 0., 0.
        col[:, 2:3, 0], col[:, 2:3, 1], col[:, 2:3, 2] = 0., 255., 0.
        mix3_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_mix3 = utils.ImageClass(col)
        mean_mix3 = np.pi / 3.
        energy_mix3_region = 3 * 4 * np.pi * np.pi / 9.

        col = color2.copy()
        col[:, :1, 0], col[:, :1, 1], col[:, :1, 2] = 255., 0., 0.
        col[:, 1:2, 0], col[:, 1:2, 1], col[:, 1:2, 2] = 255., 255., 255.
        col[:, 2:3, 0], col[:, 2:3, 1], col[:, 2:3, 2] = 0., 255., 0.
        mix4_coordinates = np.array([[i, j] for i in np.arange(col.shape[0]) for j in np.arange(col.shape[1])])
        im_mix4 = utils.ImageClass(col)
        mean_mix4 = np.pi / 3.
        energy_mix4_region = 3 * 4 * np.pi * np.pi / 9.

        self.expected_mix_mean = (
            ((mix1_coordinates, im_mix1.image), mean_mix1),
            ((mix2_coordinates, im_mix2.image), mean_mix2),
            ((mix3_coordinates, im_mix3.image), mean_mix3),  # Check that white does not add up to mean
            ((mix4_coordinates, im_mix4.image), mean_mix4),  # Check that black does not add up to mean
        )

        self.energy_color = (
            ((mix2_coordinates, im_mix2.image), energy_mix2_region),
            ((mix3_coordinates, im_mix3.image), energy_mix3_region),
            ((mix4_coordinates, im_mix4.image), energy_mix4_region),
        )


    def test_rgb_to_green(self):
        '''
        Tests whether the rgb_to_hsi function works.
        '''
        for input_vars, h, s, i, m in self.expected_values:
            pred_h, pred_s, pred_i = energies.rgb_to_hsi(*input_vars)
            self.assertEqual(np.linalg.norm(pred_h - h) < 0.0001, True)
            self.assertEqual(np.linalg.norm(pred_s - s) < 0.0001, True)
            self.assertEqual(np.linalg.norm(pred_i - i) < 0.0001, True)


    def test_mean_color_in_region_on_homogeneously_colored_surfaces(self):
        '''
        Tests whether the mean_color_in_region function works.
        '''
        for input_vars, _, _, _, expected_angle in self.expected_values:
            predicted_angle = energies.mean_color_in_region(*input_vars)
            self.assertEqual(np.linalg.norm(expected_angle - predicted_angle) < 0.0001, True)


    def test_mean_color_in_region_on_mixed_colored_surface(self):
        '''
        Tests whether the mean_color_in_region function works on mixed colours and correctly ignores black and white.
        :return:
        '''
        for input_vars, expected_mean_angle in self.expected_mix_mean:
            predicted_mean_angle = energies.mean_color_in_region(*input_vars)
            self.assertEqual(np.linalg.norm(expected_mean_angle - predicted_mean_angle) < 0.0001, True)


    def test_hue_color_distance(self):
        '''
        :return:
        '''
        input_vars, expected_distance = self.color_pairs
        predicted_distance = energies.hue_color_distance(*input_vars)
        self.assertEqual(np.linalg.norm(expected_distance - predicted_distance) < 0.0001, True)


    def test_mean_color_energy_per_region(self):
        for input_vars, expected_energy in self.energy_color:
            predicted_energy = energies.mean_color_energy_per_region(*input_vars)
            self.assertEqual(np.linalg.norm(expected_energy - predicted_energy) < 0.0001, True)


if __name__ == '__main__':
    unittest.main()