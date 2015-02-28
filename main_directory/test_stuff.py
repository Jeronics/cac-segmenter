__author__ = 'jeroni'
import utils
import cv
import cv2
import numpy as np
import PIL

if __name__ == '__main__':
    name = 'images_tester/eagle2.png'

    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    name = 'images_tester/mask_00.png'

    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    name = 'images_tester/image_ovella.png'

    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    name = 'images_tester/superhero.png'

    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    name = 'images_tester/banana1.png'

    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    name = 'images_tester/banana1.png'
    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    print image.save_path
    image.reshape(new_height=200, new_width=400)
    print image.path, image.save_path
    image.save_image()

    # for infile in sys.argv[1:]:
    # outfile = os.path.splitext(infile)[0] + ".thumbnail"
    #     if infile != outfile:
    #         try:
    #             im = Image.open(infile)
    #             im.thumbnail(size, Image.ANTIALIAS)
    #             im.save(outfile, "JPEG")
    #         except IOError:
    #             print "cannot create thumbnail for '%s'" % infile

    #
    #
    # contours = np.loadtxt('images_tester/contour_test.txt', float)
    # cont = [contours.astype(int)]
    # im = image.image
    #
    # cnt = cont[0]
    # M = cv2.moments(cnt)
    # print M
    # # cv2.drawContours(im, [contours.astype(int)], -1, (0, 0, 0), 3)
    # # cv2.imshow('output', im)
    #
