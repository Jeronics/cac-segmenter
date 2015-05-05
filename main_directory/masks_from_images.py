import utils
from pylab import ginput, close

if __name__ == '__main__':
    file_name = 'images.txt'
    output_file_name = 'input.txt'
    with open(file_name, 'r') as f:
        input_images = f.read().split('\n')

    f = open(output_file_name, 'w')

    for image_name in input_images:
        image = utils.ImageClass()
        image.read_png(image_name)
        print("Please click")
        image.plot_image(show_plot=False)
        input_points = ginput(2)
        print("clicked", input_points)
        f.write(image_name + '\t' + str(input_points[0][0]) + '\t' + str(input_points[0][1]) + '\t' + str(input_points[1][0]) +
                '\t' + str(input_points[1][1]) + '\n')
        close()

    f.close()
