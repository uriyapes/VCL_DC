from model import NeuralNet
import parse_image_seg2 as parse_image_seg
import param_manager
import my_utilities
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
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
        json_path = os.path.join(args.params_dir, 'unitest_params1.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        self.params = param_manager.ModelParams(json_path).dict

        if self.params['random seeds'] == 1:
            self.params['tf seed'] = random.randint(1, 2 ** 31)
            self.params['np seed'] = random.randint(1, 2 ** 31)

        NeuralNet.set_seeds(int(self.params['tf seed']), int(self.params['np seed']))

    def tearDown(self):
        tf.reset_default_graph()

    def test_model_vars_after_run(self):
        args = parser.parse_args()
        json_path = os.path.join(args.params_dir, 'image_segmentation_params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        dataset_dict = param_manager.DatasetParams(json_path)

        keep_prob = 0.5
        # depth of 5
        hidden_size_list = [256, 256, 256, 256]
        dropout_hidden_list = [0, 0, 0, keep_prob]

        dataset = parse_image_seg.Dataset(dataset_dict)
        model = NeuralNet(hidden_size_list, dropout_hidden_list, dataset, self.logger, self.params)

        model.build_model()
        model.train_model()
        test_set, test_labels, network_acc, test_pred_eval = model.eval_model()
        checkpoint_path = "./results/unitest1.ckpt"
        self.compare_to_ckpt(model, checkpoint_path)



    def test_model_init(self):
        args = parser.parse_args()
        json_path = os.path.join(args.params_dir, 'image_segmentation_params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        dataset_dict = param_manager.DatasetParams(json_path)

        keep_prob = 0.5
        # depth of 5
        hidden_size_list = [256, 256, 256, 256]
        dropout_hidden_list = [0, 0, 0, keep_prob]

        dataset = parse_image_seg.Dataset(dataset_dict)
        params = self.params
        params['number of epochs'] = 0
        model = NeuralNet(hidden_size_list, dropout_hidden_list, dataset, self.logger, params)

        model.build_model()
        model.train_model()
        test_set, test_labels, network_acc, test_pred_eval = model.eval_model()
        checkpoint_path = "./results/unitest_init.ckpt"
        self.compare_to_ckpt(model, checkpoint_path)



    def compare_to_ckpt(self, model, checkpoint_path):
        with model.sess.as_default() as sess:
            reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            assert(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) == len(var_to_shape_map))
            for key in var_to_shape_map:
                self.logger.info('Check variable: ' + key)
                valid_tensor_value = reader.get_tensor(key)
                curr_tensor_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=key)[0]
                curr_tensor_value = sess.run(curr_tensor_var)
                # tf.get_tensor_by_name(key+":0") #not working for some reason, maybe version issues?
                assert(np.array_equal(curr_tensor_value, valid_tensor_value))


if __name__ == '__main__':
    unittest.main()
