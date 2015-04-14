__author__ = 'jeroni'
import unittest
import energies
import numpy as np


class TestEnergies(unittest.TestCase):
    def setUp(self):
        # Case 1
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

        # Case 2
        vertices2 = np.array([
            [0, 0],
            [2 * np.sqrt(3), 1 / 2.],
            [4, 0]
        ])
        d2 = 10
        gradients2 = np.array([
            [0, 18],
            [0, -18],
            [0, 0]
        ])
        energy2 = 81

        # Case 3
        vertices3 = np.array([
            [0, 0],
            [10, 0],
            [0, 1]
        ])
        d3 = 10
        gradients3 = np.array([
            [0, -18],
            [0, 0],
            [0, 18],
        ])
        energy3 = 81

        self.expected_values = (
            ((vertices1, d1), gradients1, energy1),
            ((vertices1+1212, d1), gradients1, energy1),
            # ((vertices2, d2), gradients2, energy2),
            ((vertices3, d3), gradients3, energy3),
        )

    def test_vertex_constraint_grad(self):
        '''
        Tests whether the vertex_constraint_grad function works.
        :return:
        '''
        for input_vars, expected, _ in self.expected_values:
            predicted = energies.grad_vertex_constraint(*input_vars)
            print '\n',predicted
            print expected
            print predicted==expected
            self.assertEqual(sum(sum(predicted==expected)), predicted.size)

    # def test_vertex_constraint(self):
    #     '''
    #     Tests whether the vertex_constraint function works.
    #     :return:
    #     '''
    #     for input_vars, _, expected in self.expected_values:
    #         predicted = energies.vertex_constraint(*input_vars)
    #         self.assertEqual(predicted, expected)

if __name__ == '__main__':
    unittest.main()