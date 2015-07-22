import os
import utils

if __name__ == '__main__':
    folder = '../../1obj/'
    depth = 1
    generator = utils.walk_level(folder, depth)
    gens = [[r, f] for r, d, f in generator]
    f = open('../../there.txt', 'w')
    for root, files in gens:
        folder2 = root + '/human_seg/'
        print folder2
        for a, b, c in os.walk(folder2):
            f.write(folder2 + c[0]+ '\n')
            break

        # + root.split('/')[-1]+'.png'
        # f.write(folder)
