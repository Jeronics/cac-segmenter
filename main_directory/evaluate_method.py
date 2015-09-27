__author__ = 'jeronicarandellsaladich'
import os

import pandas as pd
import numpy as np


def split_list(my_list, num_splits):
    len_list = len(my_list)
    len_sublist = int(len_list / float(num_splits))
    for i in xrange(len_list / len_sublist):
        yield my_list[i * len_sublist:(i + 1) * len_sublist], my_list[:i * len_sublist] + my_list[
                                                                                          (i + 1) * len_sublist:]


def calculate_scores(df):
    TP, TN, FP, FN = df[2], df[3], df[4], df[5]
    sorensen_dice = 2 * TP / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_measure = 2 * precision * recall / (precision + recall)
    jaccard = TP / (TP + FP + FN)
    return sorensen_dice, precision, recall, F1_measure, jaccard


if __name__ == '__main__':
    results_folder_name = 'segment_bsds300/'
    folder_name = results_folder_name
    first = None

    dict_methods = {}
    images_dict = {}
    only_sorensen = True
    for r, s, f in os.walk(folder_name):
        for sub in s:
            # print sub
            sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
            if os.path.exists(sorensen_dice_file):
                sorensen_dice = pd.read_csv(sorensen_dice_file, index_col=0, sep='\t', header=None)
                images_dict[sub] = sorensen_dice.to_dict().values()[0]
                if len(sorensen_dice.columns) > 1 and only_sorensen:
                    _sorensen_dice, _precision, _recall, _F1_measure, _jaccard = calculate_scores(sorensen_dice)
                    sorensen_dice = pd.DataFrame(_sorensen_dice)
                # Remove duplicates
                grouped = sorensen_dice.groupby(level=0)
                sorensen_dice = grouped.last()
                sorensen_dice.columns = [sub] if only_sorensen else [sub]
                if not first:
                    df = sorensen_dice
                    first = True
                else:
                    df = df.join(sorensen_dice, how='inner')
                # print 'Num inst.', len(sorensen_dice)
                # print 'Mean', sorensen_dice.mean().values[0]
                # print 'STD', sorensen_dice.std().values[0]
                # print '\n'
                dict_methods[sub] = {
                    'Num_inst': len(sorensen_dice),
                    'Mean': sorensen_dice.mean().values[0],
                    'STD': sorensen_dice.std().values[0],
                    # 'num_correct':len(sorensen_dice[sorensen_dice[sub]>0.8]),
                    # 'mean_correct': sorensen_dice[sorensen_dice[sub]>0.8].mean().values[0]
                }

                # df = df.fillna(-1)
                # plt.figure()
                # for sub in s:
                # sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
                # print df.columns
                # print df[sub].values
                # if os.path.exists(sorensen_dice_file):
                # plt.plot(df[sub].values)
                # plt.legend(s, loc='lower center')
                # plt.show()

    final = pd.DataFrame.from_dict(dict_methods, orient='index')

    final_images = pd.DataFrame.from_dict(images_dict)
    # print len(final_images.values)

    final_images_t = pd.DataFrame.from_dict(images_dict, orient='index')

    # final_images_t = final_images.T
    # print np.argmin(final_images['MultiMixtureGaussianCAC']), min(final_images['MultiMixtureGaussianCAC'])
    # print np.argmin(final_images['MultivariateGaussianCAC']), min(final_images['MultivariateGaussianCAC'])
    # print np.argmin(final_images_t.mean()), min(final_images_t.max())

    # print final
    print pd.DataFrame.from_dict(final_images.mean()).sort([0])
    images = final_images_t.columns.tolist()
    scores = []
    stds = []
    num_splits = 3
    for train, test in split_list(images, num_splits):
        validated_method = final_images_t[train].T.mean().argmax()
        train_score = final_images_t[train].T.mean().max()
        test_score = final_images_t[test].loc[validated_method].mean()
        test_std = final_images_t[test].loc[validated_method].std()
        print validated_method, train_score
        print test_score
        scores.append(test_score)
        stds.append(test_std)

    print 'Final mean:', np.array(scores).mean(), np.array(scores).std()

