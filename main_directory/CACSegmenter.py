from sklearn.grid_search import ParameterGrid
import pandas as pd
import utils
import os
from CageClass import CageClass
from MaskClass import MaskClass
from ImageClass import ImageClass


class CACSegmenter():
    def __init__(self, CAC, type=None, weight=None):
        self.band_size = 500
        self.k = 50
        self.d = 10
        self.other = 10
        self.parameters = {
            'num_points': [12, 14, 16],
            'ratio': [1.05, 1.1, 1.15, 1.2, 1.25],
        }
        self.type = type
        self.weight = weight
        self.CAC = CAC

    def _load_dataset(self, dataset_name):
        assert os.path.isfile(dataset_name), 'The input dataset file name is not valid!'
        dataset = pd.read_csv(dataset_name, sep='\t')
        return dataset

    def evaluate_results(self, image, cage, mask, resulting_cage, gt_mask, results_file='results_cages'):
        utils.mkdir(results_file)
        result_file = results_file + "/" + cage.save_name
        if not resulting_cage:
            print 'No convergence reached for the cac-segmenter'
        else:
            resulting_cage.save_cage(result_file)
            res_fold = results_file + "/" + 'result' + cage.spec_name.split("cage_")[-1] + '.png'
            result_mask = utils.create_ground_truth(cage, resulting_cage, mask)
            if result_mask:
                result_mask.save_image(filename=res_fold)
            if gt_mask:
                sorensen_dice_coeff = utils.sorensen_dice_coefficient(gt_mask, result_mask)
                print 'Sorensen-Dice coefficient', sorensen_dice_coeff
                return sorensen_dice_coeff


    def _load_model(self, x, parameters):
        image = ImageClass()
        image.read_png(x.image_name)
        mask = MaskClass()
        mask.from_points_and_image([x.center_x, x.center_y], [x.radius_x, x.radius_y], image)
        cage = CageClass()
        cage.create_from_points([x.center_x, x.center_y], [x.radius_x, x.radius_y], parameters['ratio'],
                                parameters['num_points'], filename='hello_test')
        gt_mask = MaskClass()
        if x.gt_name:
            gt_mask.read_png(x.gt_name)
        else:
            gt_mask = None
        return image, mask, cage, gt_mask


    def test_model(self, dataset, params, results_folder, plot_evolution=False):
        '''
        segments a group of image given a set of parameters. If the ground_truth exists it returns an evaluation
        :return resulting cages:
        '''
        utils.mkdir(results_folder)
        results_file = results_folder + '/' + 'sorensen_dice_coeff' + '.txt'
        utils.mkdir(results_folder)
        for i, x in dataset.iterrows():
            if i < 19:
                continue

            image_obj, mask_obj, cage_obj, gt_mask = self._load_model(x, params)
            print 'Start Segmentation..'
            cac_object = self.CAC(image_obj, mask_obj, cage_obj, gt_mask, type=self.type, weight=self.weight)
            result = cac_object.segment(image_obj, mask_obj, cage_obj, None, model='mean_model',
                                        plot_evolution=plot_evolution)
            print 'End Segmentation'
            if result:
                sorensen_dice_coeff = self.evaluate_results(image_obj, cage_obj, mask_obj, result, gt_mask)
                with open(results_file, 'a') as f:
                    f.write(image_obj.spec_name + '\t' + str(sorensen_dice_coeff) + '\n')
                result.save_cage(results_folder + '/' + image_obj.spec_name + '.txt')
        return 0
        # return resulting_cages, evaluation


    def _partition_dataset(self, dataset, i_th, CV):
        '''
        Divides the dataset into Train or Test based on the i_th partition
        :param dataset (pandas dataframe):
        :param i_th (int):
        :param CV (int):
        :return: Train and Test pandas dataframes
        '''
        split_points = [int(i * len(dataset) / 5.) for i in xrange(CV + 1)]
        split_points[0] = -1
        split_points[-1] = len(dataset)
        a = split_points[i_th] + 1
        b = split_points[i_th + 1]
        Test = dataset[a:b]
        Train = pd.concat([dataset[:a], dataset[b:]])
        return Train, Test


    def _find_best_model(self, dataset, CV=5):
        results_folder = 'results/'
        parameters_performance = pd.DataFrame(self.get_parameters())
        performance_df = pd.DataFrame(dtype=float)
        for i in xrange(CV):
            _, Test = self._partition_dataset(dataset, i, CV)
            performance = []
            for i, p in enumerate(self.get_parameters()):
                results_folder_p = results_folder + 'params' + i
                performance.append(self.test_model(Test, p, results_folder_p))
            # Add a column with the performance of the method on the dataset
            performance_df[str(i)] = performance
        parameters_performance['arithmetic_mean'] = performance_df.mean(axis=1)
        parameters_performance['harmonic_mean'] = len(performance_df.columns) / (1 / performance_df).sum(axis=1)
        parameters_performance = pd.concat([parameters_performance, performance_df], axis=1)
        return parameters_performance

    def _evaluate_method(self):
        return 0

    def get_parameters(self):
        '''
        Returns a list of all possible combinations of parameters.
        :return: list with a dictionary of parameter_name: value
        '''
        specific_params = list(ParameterGrid(self.parameters))
        return specific_params

    def train_model(self, input_file, CV=5):
        '''
        This function uses cross validation to lean the optimal parameters
        :return:
        '''
        dataset = self._load_dataset(input_file)
        parameter_performances = pd.DataFrame(self.get_parameters())
        self._cross_validation(dataset, CV)
