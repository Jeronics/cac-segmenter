from ImageClass import ImageClass
from MaskClass import MaskClass
import utils


if __name__ == '__main__':
    folder_name = '../../1obj'
    depth = 1
    generator = utils.walk_level(folder_name, depth)
    gens_0 = [[r, f] for r, d, f in generator if len(r.split("/")) == len(folder_name.split("/")) + depth]
    for r, f in gens_0:
        folder_name_ = r + '/human_seg'
        depth = 0
        generator = utils.walk_level(folder_name_, depth)
        gens_1 = [[r_, f_] for r_, d_, f_ in generator if len(r_.split("/")) == len(folder_name_.split("/")) + depth]
        for r_, f_ in gens_1:
            for _f_ in f_:
                filename = folder_name_ + '/' + _f_
                image = ImageClass()
                image.read_png(filename)
                # image.plot_image()
                mask = image.image
                a = [255., 0., 0.]
                mask[mask == a] = 255.
                mask[mask != a] = 0.
                m = MaskClass()
                m.mask = mask[:, :, 0]
                # m.plot_image()
                m.save_image(filename=filename)