import cac_segmenter
import numpy as np
import utils
import sys


def walk_through_dataset(root_folder, depth):
    generator = utils.walk_level(root_folder, depth)
    model = 1
    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(root_folder.split("/")) + depth]
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
            results_path = root + '/results'
            utils.mkdir(results_path)
            if len(images) > 1:
                results_path = results_path + "/" + image.spec_name
                utils.mkdir(results_path)
            if len(masks) > 1:
                results_path = results_path + "/" + mask.spec_name
            for image in images:
                if image.spec_name == 'eagle1':
                    print 'HERE'
                else:
                    continue
                for mask in masks:
                    for cage in cages:
                        result_file = results_path + "/" + cage.save_name
                        resulting_cage = cac_segmenter.cac_segmenter(image, mask, cage, None)
                        if not resulting_cage:
                            print 'No convergence reached for the cac-segmenter'
                        else:
                            utils.save_cage(resulting_cage, result_file)


if __name__ == '__main__':
    RootFolder = '../dataset'
    depth = 2
    walk_through_dataset(RootFolder, depth)