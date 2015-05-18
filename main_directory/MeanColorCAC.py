from CACSegmenter import CACSegmenter


class MeanColorCAC(CACSegmenter):
    def __init__(self):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]


if __name__ == '__main__':
    color_cac = MeanColorCAC()
    parameter_list = color_cac.get_parameters()

    dataset = color_cac._load_dataset('BSDS300_input.txt')
    color_cac.test_model(dataset, parameter_list[0])
    # color_cac.train_model('BSDS300_input.txt')