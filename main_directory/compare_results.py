import os

import pandas as pd


if __name__ == '__main__':
    # 'segment_results_alpert_1/'--> sigma=0.5
    # 'segment_results_alpert_2/'--> sigma=1.0
    # 'segment_results_alpert_3/'--> sigma=0.25
    # 'segment_results_alpert_4/'--> sigma=0.1
    # 'segment_results_alpert_prova/'--> sigma=1.0
    # 'segment_results_alpert_3/'--> sigma=0.25

    results_folder_name = 'segment_subtests/'
    folder_name = results_folder_name
    first = None

    dict_methods = {}
    images_dict = {}
    for r, s, f in os.walk(folder_name):
        for sub in s:
            # print sub
            sorensen_dice_file = folder_name + sub + '/' + 'sorensen_dice_coeff.txt'
            if os.path.exists(sorensen_dice_file):
                sorensen_dice = pd.read_csv(sorensen_dice_file, index_col=0, sep='\t', header=None)
                images_dict[sub] = sorensen_dice.to_dict().values()[0]

                # Remove duplicates
                grouped = sorensen_dice.groupby(level=0)
                sorensen_dice = grouped.last()
                sorensen_dice.columns = [sub]
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
                    'STD': sorensen_dice.std().values[0]
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
    # final_images = final_images.drop(
    #     ['san_andres_130', 'img_3803', 'bw4', 'sg_01_img_1943_tratada', 'palovigna', 'tendrils', 'cheeky_penguin',
    #      'ireland_62_bg_061502', 'snow2_004', 'yokohm060409_dyjsn191', 'sharp_image', 'matsumt060102_dyj08', 'carriage',
    #      'dscf3623', 'egret_face', '114591144943', 'img_4214', 'img_1516', 'osaka060102_dyjsn071', 'estb_img_6461_',
    #      'b7nature_trees002', 'dsc_0959', 'img_7359_copia', 'imgp2712', 'dscn2064', 'chaom38', 'europe_holiday_484',
    #      'b2pods001', 'chain98', 'b1snake001', 'img_3083_modif', 'b19objects118', 'carrigafoyle_castle_ext',
    #      '112255696146', 'imagen_072__1_', 'dscf0459', '20060319_087'])
    # final_images = final_images.drop(
    # ['palovigna', 'culzeancastle', 'osaka060102_dyjsn071', 'img_1516', 'b4nature_animals_land009', 'san_andres_130',
    #      'dscn2064', 'dscf3623', 'matsumt060102_dyj08', '110016671724', 'img_4214', 'estb_img_6461_',
    #      'ireland_62_bg_061502', 'hpim1242', 'dscn2154', 'dscn6805', 'imgp2712', 'tendrils'])
    # final_images = final_images.drop(
    #     ['kconnors030466', 'carrigafoyle_castle_ext', '112255696146', 'picture_458', 'mexico3', 'dscf3583',
    #      'dsc01236', 'europe_holiday_484', 'london_zoo3', 'hot_air_balloons_05', 'carriage', 'pict2272',
    #      'pic0203vn0105', 'b1snake001', 'bw4', 'aaa', '20060319_087', 'boy_float_lake', 'caterpiller', 'img_3083_modif',
    #      'pic1080629574', 'dscf0034_l', 'chain98', 'b7nature_trees002', 'dscf3772', 'animal_5_bg_020803',
    #      'b19objects118', 'cheeky_penguin'])

    # final_images_t = final_images.T
    # print np.argmin(final_images['MultiMixtureGaussianCAC']), min(final_images['MultiMixtureGaussianCAC'])
    # print np.argmin(final_images['MultivariateGaussianCAC']), min(final_images['MultivariateGaussianCAC'])
    # print np.argmin(final_images_t.mean()), min(final_images_t.max())

    # print final
    print final_images.mean()
    print len(final_images)
    filename = 'text_results/subtests_AlpertGBB07/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    final.to_csv(filename + 'subtests.txt')
    # print len(final_images.values)