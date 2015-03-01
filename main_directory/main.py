import cac_segmenter
import numpy as np
import utils

def walk_through_dataset(RootFolder, depth):
    generator = utils.walk_level(RootFolder, depth)

    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(RootFolder.split("/")) + depth][1:]
    print gens
    for root, files in gens:
        images = utils.get_images(files, root)
        masks = utils.get_mask(files, root)
        cages = utils.get_cages(files, root)
        if not images:
            print root, 'has no .png image'
        elif not masks:
            print root, 'has no .png mask'
        elif not cages:
            print root, 'has no .txt cages'
        else:
            utils.mkdir(root + '/results')
            for image in images:
                for mask in masks:
                    for cage in cages:
                        if len(images)>1:
                            result = root + "/results/" +image.name+"/"+mask.spec_name+'_'+cage.spec_name+'.txt'
                        rgb_image, mask_file, init_cage_file, curr_cage_file = utils.get_inputs(
                            [None, model, image, mask, cage])
                        resulting_cage = cac_segmenter.cac_segmenter(rgb_image, mask_file, init_cage_file, curr_cage_file)
                        if not resulting_cage:
                            print 'No convergence reached for the cac-segmenter'
                        else:
                            utils.save_cage(resulting_cage, result)

if __name__ == '__main__':

    RootFolder = '../dataset'
    depth = 2
    walk_through_dataset(RootFolder, depth)