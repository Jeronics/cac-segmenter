import cac_segmenter
import numpy as np
import utils
import sys
import copy


def evaluate_results(image, cage, mask, resulting_cage, files, root):
    results_folder = root + '/results'
    utils.mkdir(results_folder)
    result_file = results_folder + "/" + cage.save_name
    if not resulting_cage:
        print 'No convergence reached for the cac-segmenter'
    else:
        resulting_cage.save_cage(result_file)
        gt_mask = utils.get_ground_truth(image, files)
        res_fold = results_folder + "/" + 'result' + cage.spec_name.split("cage_")[-1] + '.png'
        result_mask = utils.create_ground_truth(cage, resulting_cage, mask)
        if result_mask:
            result_mask.save_image(filename=res_fold)
        print res_fold
        if gt_mask:
            sorensen_dice_coefficient = utils.sorensen_dice_coefficient(gt_mask, result_mask)
            print 'Sorensen-Dice coefficient', sorensen_dice_coefficient


def walk_through_dataset(root_folder, depth, start_from=False, plot_evolution=False):
    generator = utils.walk_level(root_folder, depth)
    model = 1
    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(root_folder.split("/")) + depth]
    print gens


    # This boolean controls whether the algorithm will be used on a specific image or not
    if start_from:
        go = False
        print 'Yes'
    else:
        go = True

    for root, files in gens:
        images = utils.get_images(files, root)
        masks = utils.get_masks(files, root)
        cages = utils.get_cages(files, root)
        if not images or not masks or not cages:
            if not images:
                print root, 'has no .png image'
            if not masks:
                print root, 'has no .png mask'
            if not cages:
                print root, 'has no .txt cages'
        else:
            # TODO: FIX TO ALLOW MORE IMAGES
            for image in images:
                if image.spec_name == start_from:
                    go = True
                if not go:
                    continue
                for mask in masks:
                    for cage in cages:
                        print '\nSegmenting', image.root
                        aux_cage = copy.deepcopy(cage)
                        resulting_cage = cac_segmenter.cac_segmenter(image, mask, aux_cage, None,
                                                                     plot_evolution=plot_evolution)
                        evaluate_results(image, cage, mask, resulting_cage, files, root)


if __name__ == '__main__':
    RootFolder = '../dataset'
    depth = 2
    walk_through_dataset(RootFolder, depth, start_from=False, plot_evolution=False)
