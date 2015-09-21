import numpy as np

from ImageClass import ImageClass


if __name__ == '__main__':
    filename = '../../synthetic_images/images/gaussian_image_5.png'
    image = ImageClass()
    image.read_png(filename)
    image.plot_image()
    x = np.copy(image.image)
    mu, sigma = 0, 5.0
    y = x + np.random.normal(mu, sigma, size=x.shape)
    y[y > 255.] = 255.
    y[y < 0.] = 0.
    image.image = y
    print image.image
    image.plot_image()
    filename = '../../synthetic_images/images/gaussian_image_5.png'
    image.save_image(filename=filename)
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
