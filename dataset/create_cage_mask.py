__author__ = 'jeronicarandellsaladich'

from Tkinter import *
from PIL import Image,ImageTk
import numpy as np
import png
from main_directory import utils

PI = 3.14159265358979323846264338327950288419716939937510

def create_mask_and_cage_points(c, p, image, num_cage_points, filename='output'):
    '''
    This function creates a mask and a sequence of cages.
    :param c:
    :param p:
    :param im_shape:
    :param num_cage_points:
    :return:
    '''
    im_shape = image.shape
    radius = np.linalg.norm(np.array(c) - np.array(p))
    radius_cage_ratio = [1.3, 1.5, 1.7]
    im = np.zeros(im_shape, dtype='uint8')
    print 'Shape', im_shape
    print c
    mask_points = []

    # careful im_shape is (max(y), max(x))
    for x in xrange(im_shape[1]):
        for y in xrange(im_shape[0]):
            if pow(y - c[0], 2) + pow(x - c[1], 2) <= pow(radius, 2):
                im[y, x] = 255
                mask_points.append([y, x])

    png.from_array(im, 'L').save("circle_image.png")
    print type(num_cage_points) is list
    if type(num_cage_points) is not list:
        num_cage_points = [num_cage_points]
    cages = {}
    for ratio in radius_cage_ratio:
        for n in num_cage_points:
            print n
            cage = []
            for i in xrange(0, n):
                angle = 2 * i * PI / n
                y, x = radius * ratio * np.sin(angle), radius * ratio * np.cos(angle)
                cage.append([y + c[0], x + c[1]])
            cages[str(n) + '_' + str(ratio)] = np.array(cage)
    print cages.keys()
    utils.plotContourOnImage(np.array(mask_points), image,
                             points=cages[str(num_cage_points[0]) + '_' + str(radius_cage_ratio[0])],
                             points2=cages[str(num_cage_points[1]) + '_' + str(radius_cage_ratio[1])])
    exit()


def open_canvas(File):
    root = Tk()
    out_filename = '/'.join(File.split('/')[:-1])
    print out_filename

    # File = '../dataset'
    # for root, dirs, files in utils.walk_level(File, 1):
    # print root

    img = ImageTk.PhotoImage(Image.open(File))
    image = utils.read_png(File)

    # setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=0, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E + W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N + S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set, width=image.shape[1],
                    height=image.shape[0])
    canvas.grid(row=0, column=0, sticky=N + S + E + W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH, expand=1)

    print File
    canvas.create_image(0, 0, image=img, anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    text_file = open("Mask_output.txt", "w")

    num_cage_points = [8, 9, 10]

    def printcoords(event):
        # reading the center and a point on the radius in order to create a mask.
        text_file.write("%.8e\t%.8e\n" % (event.y, event.x))
        global COUNTER
        global CENTER
        global RADIUS_POINT
        if COUNTER == 0:
            # The first point is the center
            CENTER = [event.y, event.x]
            print 'Center', CENTER
            COUNTER += 1
        else:
            # The second point is a point in the radius
            RADIUS_POINT = [event.y, event.x]
            print 'RADIUS_POINT', RADIUS_POINT
            create_mask_and_cage_points(CENTER, RADIUS_POINT, image, num_cage_points, filename=out_filename)

    # mouseclick event
    canvas.bind("<Button 1>", printcoords)

    root.mainloop()
    text_file.close()

if __name__ == '__main__':
    File = '../test/ovella/image_ovella.png'
    COUNTER = 0
    # function to be called when mouse is clicked
    CENTER = []
    RADIUS_POINT = []
    open_canvas(File)