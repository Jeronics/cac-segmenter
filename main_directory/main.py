import cac_segmenter
import numpy as np
import utils

if __name__ == '__main__':

    RootFolder = '../dataset'
    depth = 2
    generator = utils.walk_level(RootFolder, depth)

    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(RootFolder.split("/")) + depth][1:]
    print gens
    for root, files in gens:
        # function to be called when mouse is clicked
        model, image, mask, cage = 0, '', '', ''
        for f in files:
            if f.split('.')[-1] == 'png' and f.split("/")[-1].split("_")[0] != 'mask':
                print f
                model = 1
                image = root + "/" + f
        for f in files:
            if f.split('.')[-1] == 'png' and f.split("/")[-1].split("_")[0] == 'mask':
                mask = root + "/" + f
        if image == '' or mask == '':
            print root, 'has nothing'
        else:
            utils.mkdir(root + '/results')
            for f in files:
                if f.split('.')[-1] == 'txt' and f.split("/")[-1].split("_")[0] == 'cage':
                    cage = root + "/" + f
                    result = root + "/results/" + f.split('.txt')[0]
                    rgb_image, mask_file, init_cage_file, curr_cage_file = utils.get_inputs(
                        [None, model, image, mask, cage])
                    resulting_cage = cac_segmenter.cac_segmenter(rgb_image, mask_file, init_cage_file, curr_cage_file)
                    if resulting_cage == None:
                        print 'No convergence reached for the cac-segmenter'
                    else:
                        utils.save_cage(resulting_cage, result)