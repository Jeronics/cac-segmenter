__author__ = 'jeroni'
# import utils
# import cv
# import cv2
# import numpy as np
# import PIL
#
# if __name__ == '__main__':
#     name = 'images_tester/eagle2.png'
#
#     image = utils.ImageClass()
#     image.read_png(name)
#     print image.path
#     print image.shape
#     print image.image.shape
#     image.plot_image(show_plot=False)
#
#     name = 'images_tester/mask_00.png'
#
#     image = utils.ImageClass()
#     image.read_png(name)
#     print image.path
#     print image.shape
#     print image.image.shape
#     image.plot_image(show_plot=False)
#
#     name = 'images_tester/image_ovella.png'
#
#     image = utils.ImageClass()
#     image.read_png(name)
#     print image.path
#     print image.shape
#     print image.image.shape
#     image.plot_image(show_plot=False)
#
#     name = 'images_tester/superhero.png'
#
#     image = utils.ImageClass()
#     image.read_png(name)
#     print image.path
#     print image.shape
#     print image.image.shape
#     image.plot_image(show_plot=False)
#
#     name = 'images_tester/banana1.png'
#
#     image = utils.ImageClass()
#     image.read_png(name)
#     print image.path
#     print image.shape
#     print image.image.shape
#     image.plot_image(show_plot=False)
#
#     name = 'images_tester/banana1.png'
#     image = utils.ImageClass()
#     image.read_png(name)
#     print image.path
#     print image.shape
#     print image.image.shape
#     image.plot_image(show_plot=False)
#
#     print image.save_path
#     image.reshape(new_height=200, new_width=400)
#     print image.path, image.save_path
#     image.save_image()
#
#     # import the necessary packages
#     import numpy as np
#     import argparse
#     import imutils
#     import glob
#     import cv2
#
# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
import utils

if __name__ == '__main__':
    name = 'images_tester/eagle2.png'

    image = utils.ImageClass()
    image.read_png(name)
    print image.path
    print image.shape
    print image.image.shape
    image.plot_image(show_plot=False)

    # load the image image, convert it to grayscale, and detect edges
    template = cv2.imread(image.path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    # cv2.imshow("Template", template)
    ret,thresh = cv2.threshold(template,152,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print len(contours)
    print type(template),(tH, tW)
    print template
    for c in contours:
        cv2.drawContours(template, [c], 0, 255,-1)
    gt_mask = utils.MaskClass(template, image.save_path)
    gt_mask.save_mask()


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
