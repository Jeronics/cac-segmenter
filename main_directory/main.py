__author__ = 'jeronicarandellsaladich'

import sys
import time
from utils import *


#Just measuring time!
start=time.time()



#./cac model imatge mascara caixa_init [caixa_curr]
if (len(sys.argv) != 6 and len(sys.argv) != 5 ):
    print 'Wrong Use!!!! Expected Input ' +sys.argv[0] + ' model(int) image(int) mask(int) init_cage(int) [curr_cage(int)]'
    sys.exit(1)

for arg in sys.argv:
    print arg
model = sys.argv[0]
mask = int(sys.argv[1])
init_cage= int(sys.argv[2])
if len(sys.argv) == 6:
    curr_cage = int(sys.argv[5]);
else:
    curr_cage = None


# PATHS
test_path = r'../test/elefant/'
mask_num = '%(number)02d' % {"number": mask}
init_cage_name = '%(number)02d' % {"number": init_cage}
curr_cage_name = '%(number)02d' % {"number": curr_cage}

mask_name = test_path + 'mask_'+mask_num+'.pgm'
init_cage_name = test_path + 'cage_'+init_cage_name+'.txt'
curr_cage_name = test_path + 'cage_'+curr_cage_name+'.txt'

# LOAD Cage and Mask

mask_file = read_pgm(mask_name,byteorder='>')

mask_file = read_pgm(mask_name,byteorder='>')
init_cage_file = np.loadtxt(init_cage_name, float)
curr_cage_file = np.loadtxt(curr_cage_name, float)

print init_cage_file
print mask_file






# THE END



#Just measuring time!
end=time.time()
print end-start