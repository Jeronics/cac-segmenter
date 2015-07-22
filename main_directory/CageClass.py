import numpy as np


class CageClass:
    def __init__(self, cage=np.array([]), filename=''):
        self.cage = cage
        self.shape = cage.shape
        self.num_points = len(cage)
        self.path = filename
        self.name = filename.split("/")[-1]
        self.spec_name = self.name.split('.txt')[0]
        self.root = "/".join(filename.split("/")[:-1]) + "/"
        self.save_name = self.spec_name + '_out.txt'

    def read_txt(self, filename):
        cage = np.loadtxt(filename, float)
        # Rotate the cage to (y,x)
        rot = np.array([[0, 1], [1, 0]])
        cage = np.dot(cage, rot)
        self.__init__(cage, filename)

    def create_from_points(self, c, p, ratio, num_cage_points, filename=''):
        '''
        This function instantiates a cage.
            The cages are created clockwise from (x,y)=( c_x + r*R, c_y)
        '''
        radius = np.linalg.norm(np.array(c) - np.array(p))
        cage = []
        for i in xrange(0, num_cage_points):
            angle = 2 * i * np.pi / num_cage_points
            x, y = radius * ratio * np.sin(angle), radius * ratio * np.cos(angle)
            cage.append([y + c[1], x + c[0]])
        self.__init__(cage=np.array(cage), filename='')
        return cage

    def save_cage(self, filename):
        text_file = open(filename, "w")

        for x, y in self.cage:
            # Un-Rotate to (y,x)
            text_file.write("%.8e\t%.8e\n" % (y, x))