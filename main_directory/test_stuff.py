import numpy as np

from ImageClass import ImageClass
import pandas as pd
from MaskClass import MaskClass
if __name__ == '__main__':
    # Resize Image
    # pear=MaskClass()
    # pear.read_png('../../morphing_gt/mask_pear1.png')
    # pear.reshape(new_height=400)
    # pear.save_image(filename='../../morphing_gt/mask_pear1.png')

    # # Find average F-measure
    # filename_subset = '../polygon_tools/subset.txt'
    # filename_expanded_subset = '../polygon_tools/expanded-subset.txt'
    # filename_gbp_ucm = '../polygon_tools/gBp-ucm.txt'
    # filename_nips2012 = '../polygon_tools/NIPS2012.txt'
    # subset = pd.read_csv(filename_subset, names=['id'])
    # gbp = pd.read_csv(filename_gbp_ucm, header=None, names=['id', 'F'])
    # nips = pd.read_csv(filename_nips2012, header=None, names=['id', 'F'])
    # subset_values = subset['id'].values
    # print gbp.loc[gbp['id'].isin(subset_values)].mean()
    # print gbp.loc[gbp['id'].isin(subset_values)].std()
    # print nips.loc[nips['id'].isin(subset_values)].mean()
    # print nips.loc[nips['id'].isin(subset_values)].std()
    #
    # GBP=pd.DataFrame(gbp.loc[gbp['id'].isin(subset_values)]).set_index('id')
    # NIPS=pd.DataFrame(nips.loc[nips['id'].isin(subset_values)]).set_index('id')
    # print GBP.std()
    # print NIPS.std()

    # # Turn grayer
    filename= '../../TesinaFinalJeroni/images/image_in_question.png'
    image = ImageClass()
    image.read_png(filename)
    im=image.image.copy()
    boolean = im < np.ones(im.shape)*10
    im[boolean] = np.ones(im.shape)[boolean]*80.
    image.image = im
    filename_save = '../../TesinaFinalJeroni//images/image_in_question_lighter_2.png'
    image.save_image(filename_save)

    # # apply noise
    # filename = '../../synthetic_images/images/gaussian_image_4.png'
    # image = ImageClass()
    # image.read_png(filename)
    # image.plot_image()
    # x = np.copy(image.image)
    # mu, sigma = 0, 5.0
    # y = x + np.random.normal(mu, sigma, size=x.shape)
    # y[y > 255.] = 255.
    # y[y < 0.] = 0.
    # image.image = y
    # print image.image
    # image.plot_image()
    # filename = '../../synthetic_images/images/gaussian_image_4_noise.png'
    # image.save_image(filename=filename)

    # # other
    # depth = 1
    # generator = utils.walk_level(folder, depth)
    # gens = [[r, f] for r, d, f in generator]
    # f = open('../../there.txt', 'w')
    # for root, files in gens:
    # folder2 = root + '/human_seg/'
    # for root, files in gens:
    # folder2 = root + '/human_seg/'
    # print folder2
    # for a, b, c in os.walk(folder2):
    #         f.write(folder2 + c[0]+ '\n')
    #         break
    #
    #     # + root.split('/')[-1]+'.png'
    #     # f.write(folder)
