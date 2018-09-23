import os
import json
import numpy as np
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument('--datasets_dir', default='./datasets', help="directory containing .json file detailing the dataset params")

# TODO: add input check for parameters
class Params(object):
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self):
        pass
        # self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        # with open(json_path, 'w') as f:
        #     json.dump(self.__dict__, f, indent=4)
        save_dict_to_json(self.__dict__, json_path)

    def update(self, json_path):
        """Loads parameters from json file and update the Params class with those parameters"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    # @classmethod
    # def create_json_param_file(cls, json_path):
    #     dict_params = cls._create_default_model_params()
    #     save_dict_to_json(dict_params, json_path)

    # @classmethod
    # def _create_default_model_params(cls):
    #     pass


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array)
        # d = {k: float(v) for k, v in d.items()}
        for k,v in d.items():
            # if type(v) != str and v is not None and type(v) != int:
            #     v = float(v)
            d[k] = v
        json.dump(d, f, indent=4)

class ModelParams(Params):
    def __init__(self, batch_norm=False, keep_prob_list=[0, 0, 0, 0.5], hidden_size_list=[256, 256, 256, 256],
                 activation='RELU', use_vcl=False, vcl_gamma=0.01, vcl_sample_size=10, l2_loss_coeff=0.0001,
                 random_seeds_flag=1, tf_seed=230, np_seed=100, num_of_epochs=500, ckpt_flag=0, ckpt_file_name=None):
        super(ModelParams, self).__init__()
        self.dict['batch norm'] = batch_norm
        self.dict['dropout keep prob list'] = keep_prob_list
        self.dict['hidden size list'] = hidden_size_list
        self.dict['activation'] = activation
        self.dict['vcl'] = use_vcl
        self.dict['gamma'] = vcl_gamma
        self.dict['vcl sample size'] = vcl_sample_size
        self.dict['l2 coeff'] = l2_loss_coeff
        self.dict['random seeds'] = random_seeds_flag
        self.dict['tf seed'] = tf_seed
        self.dict['np seed'] = np_seed
        self.randomize_seeds()
        self.dict['number of epochs'] = num_of_epochs
        self.dict['check point flag'] = ckpt_flag
        self.dict['check point name'] = ckpt_file_name

    def randomize_seeds(self):
        if self.dict['random seeds']:
            self.dict['tf seed'] = random.randint(1, 2 ** 31)
            self.dict['np seed'] = random.randint(1, 2 ** 31)
        else:
            assert type(self.dict['tf seed']) == int and type(self.dict['np seed']) == int


class DatasetParams(Params):
    def __init__(self, dataset_name='image-segmentation', filename_labels='labels_py.dat',
                            filename_indexes_test='folds_py.dat', filename_validation_indexes='validation_folds_py.dat',
                            assert_values_flag=True, label_encode_one_hot=True):
        super(DatasetParams, self).__init__()
        args = parser.parse_args()
        folder_path = os.path.join(args.datasets_dir, dataset_name)
        self.dict['dataset_name'] = dataset_name
        filename_data = dataset_name + "_py.dat"
        self.dict['FILENAME_DATA'] = os.path.join(folder_path, filename_data)
        self.dict['FILENAME_LABELS'] = os.path.join(folder_path, filename_labels)
        self.dict['FILENAME_INDEXES_TEST'] = os.path.join(folder_path, filename_indexes_test)
        self.dict['FILENAME_VALIDATION_INDEXES'] = os.path.join(folder_path, filename_validation_indexes)
        self.dict['assert_values_flag'] = assert_values_flag
        self.dict['validation_train_ratio'] = 5.0
        self.dict['test_alldata_ratio'] = 300.0 / 330
        self.dict['fold'] = 1
        self.dict['label_encode_one_hot'] = label_encode_one_hot


def gen_param_files():
    # Test save_dict_to_json function
    json_path = os.path.join("./Params", 'json')
    d = {'a': 3, 'c': 'hello'}
    save_dict_to_json(d, json_path)

    # Test the params class
    params = Params()
    params.b = 2
    params.save(json_path)

    params.save(json_path + "2")

    # Create template for model params json file
    json_path_template = os.path.join("./Params", 'model_params_template.json')
    model_params = ModelParams()
    dict_params = model_params.dict
    model_params.save(json_path_template)
    model_params.update(json_path_template)
    assert dict_params == model_params.dict
    # save_dict_to_json(dict_params, json_path_template)

    model_params.dict['number of epochs'] = 5
    model_params.dict['random seeds'] = 0
    model_params.dict['tf seed'] = 230
    model_params.dict['np seed'] = 100
    json_path = os.path.join("./Params", 'unitest_params1.json')
    model_params.save(json_path)


    model_params.update(json_path_template)
    model_params.dict['vcl'] = 1
    json_path = os.path.join("./Params", 'vcl.json')
    model_params.save(json_path)

    model_params.update(json_path_template)
    model_params.dict['dropout keep prob list'] = [1, 1]
    model_params.dict['hidden size list'] = [300, 100]
    model_params.dict['number of epochs'] = 91
    json_path = os.path.join("./Params", 'lenet_300_100.json')
    model_params.save(json_path)



    # Create image segmentation dataset params
    json_path = os.path.join("./Params", 'image_segmentation_params.json')
    image_segmentation_params = DatasetParams()
    image_segmentation_values = image_segmentation_params.dict.values()
    image_segmentation_params.save(json_path)
    image_segmentation_params.update(json_path)
    assert image_segmentation_values == image_segmentation_params.dict.values()
    print image_segmentation_params.dict
    image_segmentation_params.dict['dataset_name'] = 'bad name'
    assert image_segmentation_values != image_segmentation_params.dict.values()

    # Create dataset params
    json_path = os.path.join("./Params", 'abalone.json')
    abalone_params = DatasetParams(dataset_name='abalone', assert_values_flag=False)
    abalone_params.save(json_path)

    json_path = os.path.join("./Params", 'contrac.json')
    contrac_params = DatasetParams(dataset_name='contrac', assert_values_flag=False)
    contrac_params.save(json_path)

    json_path = os.path.join("./Params", 'car.json')
    car_params = DatasetParams(dataset_name='car', assert_values_flag=False)
    car_params.save(json_path)




if __name__ == '__main__':
    # pass
    gen_param_files()
