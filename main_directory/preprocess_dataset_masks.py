import ImageClass
import MaskClass

if __name__ == '__main__':
    folder_name = '../../1obj/100_0109/human_seg/'
    mask_file = '100_0109_7.png'
    filename = folder_name + mask_file
    image = ImageClass()
    image.read_png(filename)
    image.plot_image()
    m = MaskClass()
