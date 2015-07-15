from main_directory.MaskClass import MaskClass
from main_directory.ImageClass import ImageClass

if __name__ == '__main__':
    folder_name = '../1obj/100_0109/human_seg/'
    mask_file = '100_0109_7.png'
    image = ImageClass()
    image.read_png(mask_file)
    image.plot_image()
    filename = folder_name + mask_file
    m = MaskClass()