import cac_segmenter
import numpy as np
import utils
import sys
import copy


def walk_through_dataset(root_folder, depth):
    generator = utils.walk_level(root_folder, depth)
    model = 1
    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(root_folder.split("/")) + depth]
    print gens
    for root, files in gens:
        images = utils.get_images(files, root)
        masks = utils.get_masks(files, root)
        cages = utils.get_cages(files, root)
        if not images:
            print root, 'has no .png image'
        elif not masks:
            print root, 'has no .png mask'
        elif not cages:
            print root, 'has no .txt cages'
        else:
            results_folder = root + '/results'
            utils.mkdir(results_folder)
            # TODO: FIX TO ALLOW MORE IMAGES
            # if len(images) > 1:
            # results_folder = results_folder + "/" + image.spec_name
            # utils.mkdir(results_folder)
            # if len(masks) > 1:
            #     results_folder = results_folder + "/" + mask.spec_name
            for image in images:
                for mask in masks:
                    for cage in cages:
                        print '\nSegmenting', image.root
                        result_file = results_folder + "/" + cage.save_name
                        aux_cage = copy.deepcopy(cage)
                        resulting_cage = cac_segmenter.cac_segmenter(image, mask, aux_cage, None)
                        if not resulting_cage:
                            print 'No convergence reached for the cac-segmenter'
                        else:
                            utils.save_cage(resulting_cage, result_file)
                            gt_mask = utils.get_ground_truth(image, files)
                            res_fold = results_folder + "/" + 'result' + cage.spec_name.split("cage_")[-1] + '.png'
                            result_mask = utils.create_ground_truth(cage, resulting_cage, mask)
                            result_mask.save_image(filename=res_fold)
                            print res_fold
                            if gt_mask:
                                sorensen_dice_coefficient = utils.sorensen_dice_coefficient(gt_mask, result_mask)
                                print 'Sorensen-Dice coefficient', sorensen_dice_coefficient


if __name__ == '__main__':
    RootFolder = '../dataset'
    depth = 2
    walk_through_dataset(RootFolder, depth)