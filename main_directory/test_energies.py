__author__ = 'jeroni'
import unittest
import energies
import numpy as np


class TestEnergies(unittest.TestCase):
    '''

    '''
    def setUp(self):
        '''

        :return:
        '''
        # Case 1

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
        case 2: equilateral triangle

                         * (2,2*sqrt(2)

                (0,0) *     * (4,0)

        '''
        vertices2 = np.array([
            [0, 0],
            [2, 2*np.sqrt(3)],
            [4, 0]
        ])
        d2 = 10
        dist_1_2=2*(10-2*np.sqrt(3)/(2*np.sqrt(3)-10)
        gradients2 = np.array([
            [-np.sqrt(12)-12, -np.sqrt(12)],
            [np.sqrt(12), np.sqrt(12)],
            [12+*2), 0]
        ])
        energy2 = 81

        '''
            Case 3: Slanted Isosceles triangle

                      * (2*sqrt(2),2*sqrt(2))

            (0,0) *     * (4,0)

        '''
        vertices3 = np.array([
            [0, 0],
            [2*np.sqrt(2), 2*np.sqrt(2)],
            [4, 0]
        ])
        d3 = 10
        gradients3 = np.array([
            [0, 18],
            [0, -18],
            [0, 0]
        ])
        energy3 = 81

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
            ((vertices1+1212, d1), gradients1, energy1),
            # ((vertices2, d2), gradients2, energy2),
            ((vertices3, d3), gradients3, energy3),
            ((vertices4, d4), gradients4, energy4),
        )

    def test_vertex_constraint_grad(self):
        '''
        Tests whether the vertex_constraint_grad function works.
        :return:
        '''
        for input_vars, expected, _ in self.expected_values:
            predicted = energies.grad_vertex_constraint(*input_vars)
            print '\n'
            print predicted
            print expected
            print predicted==expected
            print np.linalg.norm(predicted-expected)
            self.assertEqual(np.linalg.norm(predicted-expected) < 0.0001, True)

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