from CACSegmenter import CACSegmenter

class MeanColorCAC(CACSegmenter):
    def __init__(self):
        CACSegmenter.__init__(self)
        self.energy = 1
        self.parameters['other'] = [1, 2, 3, 4]

if __name__=='__main__':
    color_cac = MeanColorCAC()
    color_cac._load_dataset('testing_files/input_1.txt')