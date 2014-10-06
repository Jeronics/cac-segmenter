__author__ = 'jeronicarandellsaladich'
import sys

#./cac model imatge mascara caixa_init [caixa_curr]
if (len(sys.argv) != 6 and len(sys.argv) != 5 ):
    print 'Wrong Use!!!! Expected Input ' +sys.argv[0] + ' model image mask init_cage [curr_cage]'
    sys.exit(1)

for arg in sys.argv:
    print arg
model = sys.argv[0]
mask = int(sys.argv[1])
init_cage= int(sys.argv[2])
if len(sys.argv) == 6:
    curr_cage = sys.argv[5];
else:
    curr_cage = None


# PATHS
test_path = r'../../codi_enviat/test/elefant/'
mask_num = '%(number)02d' % {"number": mask}
init_cage_name = '%(number)02d' % {"number": init_cage}
curr_cage_name = '%(number)02d' % {"number": curr_cage}

mask_name = test_path + 'mask_'+mask_num+'.pgm'
init_cage_name = test_path + 'cage_'+init_cage_name+'.txt'
curr_cage_name = test_path + 'cage_'+curr_cage_name+'.txt'

# LOAD Cage and Mask
mask_file = open(mask_name, 'r')
init_cage_file = open(init_cage_name, 'r')
curr_cage_file = open(curr_cage_name, 'r')
