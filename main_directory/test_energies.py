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


if __name__ == '__main__':
    unittest.main()