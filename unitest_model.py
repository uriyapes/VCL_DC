from model import NeuralNet
import param_manager
import my_utilities

import argparse
import unittest
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--params_dir', default='./Params', help="directory containing .json file detailing the model params")

class TestModel(unittest.TestCase):
    def setUp(self):
        self.logger = my_utilities.set_a_logger('log', dirpath="./Logs", filename='unitest_logger.log')
        # Load the parameters from json file
        args = parser.parse_args()
        # json_path = os.path.join(args.params_dir, 'model_params_template.json')
        json_path = os.path.join(args.params_dir, 'unitest_params1.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = param_manager.ModelParams(json_path)

        if params.dict['random seeds'] == 1:
            params.dict['tf seed'] = random.randint(1, 2 ** 31)
            params.dict['np seed'] = random.randint(1, 2 ** 31)

        NeuralNet.set_seeds(int(params.dict['tf seed']), int(params.dict['np seed']))

    def test_model_vars(self):
        FILENAME_DATA = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/image-segmentation_py.dat'
        FILENAME_LABELS = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/labels_py.dat'
        FILENAME_INDEXES_TEST = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/folds_py.dat'
        FILENAME_VALIDATION_INDEXES = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/validation_folds_py.dat'
        assert_values_flag = True
        dataset_dict = {'name': 'image_segmentation',
                        'file_names': (
                        FILENAME_DATA, FILENAME_LABELS, FILENAME_INDEXES_TEST, FILENAME_VALIDATION_INDEXES),
                        'assert_values_flag': assert_values_flag,
                        'validation_train_ratio': 5.0,
                        'test_alldata_ratio': 300.0 / 330}

        # TODO: add support for different dropout rates in different layers
        keep_prob = 0.5
        # depth of 5
        hidden_size_list = [256, 256, 256, 256]
        dropout_hidden_list = [0, 0, 0, keep_prob]

        model = NeuralNet(hidden_size_list, dropout_hidden_list, self.logger)
        model.get_dataset(dataset_dict)

        model.build_model()
        model.train_model()
        test_set, test_labels, network_acc, test_pred_eval = model.eval_model()




if __name__ == '__main__':
    unittest.main()