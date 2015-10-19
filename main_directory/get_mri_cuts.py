import numpy as np
import nibabel as nib

from ImageClass import ImageClass
from MaskClass import MaskClass


filename = '../../Data/'


def get_cuts_from_gt(folder_dataset, filename):
    png_filename = filename.split('.')[0] + '.png'
    image_filename = folder_dataset + 'wInput/' + filename
    mask_filename = folder_dataset + 'wGT_R/' + filename
    im_save_folder = folder_dataset + 'png_im/'
    mk_save_folder = folder_dataset + 'png_im_gt/'
    img = nib.load(image_filename)
    msk = nib.load(mask_filename)
    volume_img = img.get_data()
    volume_msk = msk.get_data() * 255.
    domain = []
    for i in xrange(volume_msk.shape[0]):
        if np.max(volume_msk[i, :, :]) > 0:
            domain.append(i)
    print domain

    for c in domain[2:-2]:
        image_cut = volume_img[c, :, :].astype(np.uint8)
        mask_cut = volume_msk[c, :, :].astype(np.uint8)
        im = ImageClass(im=image_cut)
        mk = MaskClass(mask=mask_cut)

        name = 'c_' + str(c) + '_'

        im_save_filename = im_save_folder + name + png_filename
        mk_save_filename = mk_save_folder + name + png_filename

        im.plot_image()
        im.save_image(filename=im_save_filename)
        mk.plot_image()
        im.save_image(filename=mk_save_filename)


if __name__ == '__main__':
    folder_dataset = '../../Data/'
    filename = 'wSubject_1.nii'
    get_cuts_from_gt(folder_dataset, filename)
