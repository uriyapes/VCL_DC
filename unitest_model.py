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
        self.model_params = param_manager.ModelParams()
        self.model_params.update(json_path)
        self.params = self.model_params.dict

        NeuralNet.set_seeds(self.params['tf seed'], self.params['np seed'])

    def tearDown(self):
        tf.reset_default_graph()


    def test_model_vars_after_run(self):
        args = parser.parse_args()
        json_path = os.path.join(args.params_dir, 'image_segmentation_params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        dataset_params = param_manager.DatasetParams()
        dataset_params.update(json_path)
        dataset_dict = dataset_params.dict
        dataset = parse_image_seg.Dataset(dataset_dict)
        checkpoint_path = "./results/unitest2.ckpt"
        model = NeuralNet(dataset, self.logger, self.params)
        with model:
            model.build_model()
            model.train_model()
            self.compare_to_ckpt(model, checkpoint_path)



    def test_model_init(self):
        args = parser.parse_args()
        json_path = os.path.join(args.params_dir, 'image_segmentation_params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        dataset_params = param_manager.DatasetParams()
        dataset_params.update(json_path)
        dataset_dict = dataset_params.dict

        dataset = parse_image_seg.Dataset(dataset_dict)
        params = self.params
        params['number of epochs'] = 0
        checkpoint_path = "./results/unitest_init.ckpt"
        model = NeuralNet(dataset, self.logger, params)
        with model:
            model.build_model()
            model.train_model()
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
