__author__ = 'jeroni'
import utils


def get_names_of_files_in_folder(root_folder):
    depth = 0
    generator = utils.walk_level(root_folder, depth)
    gens = [[r, f] for r, d, f in generator if len(r.split("/")) == len(root_folder.split("/")) + depth]
    all_files = []
    for root, files in gens:
        all_files.append(files)
    return all_files[0]


def list_intersection(a, b):
    return list(set(a) & set(b))


def list_union(a, b):
    return list(set(a) | set(b))


if __name__ == '__main__':
    filename = '../../BSDS300/images/test'
    test_files = get_names_of_files_in_folder(filename)
    filename = '../../BSDS300/images/train'
    train_files = get_names_of_files_in_folder(filename)
    input_images = list_union(test_files, train_files)
    filename = '../../gt_images'
    gt_images = get_names_of_files_in_folder(filename)
    images_with_gt = list_intersection(input_images, gt_images)
    print images_with_gt