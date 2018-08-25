import os
import json
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_dir', default='./datasets', help="directory containing .json file detailing the dataset params")


class Params(object):
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    @classmethod
    def create_json_param_file(cls, json_path):
        dict_params = cls._create_default_model_params()
        save_dict_to_json(dict_params, json_path)

    @classmethod
    def _create_default_model_params(cls):
        pass


class ModelParams(Params):
    def __init__(self, json_path):
        super(ModelParams, self).__init__(json_path)

    @classmethod
    def _create_default_model_params(cls):
        return cls.create_model_params()

    @classmethod
    def create_model_params(cls, batch_norm=False, keep_prob=0.5, num_of_layers=4, activation='RELU', use_vcl=False, vcl_gamma=0.01,
                            random_seeds_flag=1, tf_seed=230, np_seed=100, num_of_epochs=500, ckpt_flag=0, ckpt_file_name=None):
        dict = {}
        dict['batch norm'] = batch_norm
        dict['dropout keep prob'] = keep_prob
        dict['depth'] = num_of_layers
        dict['activation'] = activation
        dict['vcl'] = use_vcl
        dict['gamma'] = vcl_gamma
        dict['random seeds'] = random_seeds_flag
        if random_seeds_flag:
            dict['tf seed'] = random.randint(1, 2 ** 31)
            dict['np seed'] = random.randint(1, 2 ** 31)
        else:
            dict['tf seed'] = tf_seed
            dict['np seed'] = np_seed
        dict['number of epochs'] = num_of_epochs
        dict['check point flag'] = ckpt_flag
        dict['check point name'] = ckpt_file_name
        return dict

class DatasetParams(Params):
    def __init__(self, json_path):
        super(DatasetParams, self).__init__(json_path)

    @classmethod
    def _create_default_model_params(cls):
        return cls.create_model_params()

    @classmethod
    def create_model_params(cls, dataset_name='image-segmentation', filename_labels='labels_py.dat',
                            filename_indexes_test='folds_py.dat', filename_validation_indexes='validation_folds_py.dat',
                            assert_values_flag=True, label_encode_one_hot=True):
        args = parser.parse_args()
        dict = {}
        folder_path = os.path.join(args.datasets_dir, dataset_name)
        dict['dataset_name'] = dataset_name
        filename_data = dataset_name + "_py.dat"
        dict['FILENAME_DATA'] = os.path.join(folder_path, filename_data)
        dict['FILENAME_LABELS'] = os.path.join(folder_path, filename_labels)
        dict['FILENAME_INDEXES_TEST'] = os.path.join(folder_path, filename_indexes_test)
        dict['FILENAME_VALIDATION_INDEXES'] = os.path.join(folder_path, filename_validation_indexes)
        dict['assert_values_flag'] = assert_values_flag
        dict['validation_train_ratio'] = 5.0
        dict['test_alldata_ratio'] = 300.0 / 330
        dict['fold'] = 1
        dict['label_encode_one_hot'] = label_encode_one_hot
        return dict


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

def gen_param_files():
    # Test save_dict_to_json function
    json_path = os.path.join("./Params", 'json')
    d = {'a': 3, 'c': 'hello'}
    save_dict_to_json(d, json_path)

    # Test the params class
    params = Params(json_path)
    params.b = 2
    params.save(json_path)

    params.save(json_path + "2")

    # Create template for model params json file
    json_path_template = os.path.join("./Params", 'model_params_template.json')
    dict_params = ModelParams._create_default_model_params()
    save_dict_to_json(dict_params, json_path_template)
    model_params = ModelParams(json_path_template)
    print model_params.dict

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
    model_params.dict['activation'] = 'SELU'
    model_params.dict['dropout keep prob'] = 0.95
    json_path = os.path.join("./Params", 'selu.json')
    model_params.save(json_path)

    # Create image segmentation dataset params
    json_path = os.path.join("./Params", 'image_segmentation_params.json')
    DatasetParams.create_json_param_file(json_path)
    image_segmentation_params = DatasetParams(json_path)
    print image_segmentation_params.dict

    # Create image segmentation dataset params
    json_path = os.path.join("./Params", 'abalone.json')
    abalone_params = DatasetParams.create_model_params(dataset_name='abalone', assert_values_flag=False)
    save_dict_to_json(abalone_params, json_path)

    json_path = os.path.join("./Params", 'contrac.json')
    abalone_params = DatasetParams.create_model_params(dataset_name='contrac', assert_values_flag=False)
    save_dict_to_json(abalone_params, json_path)

    json_path = os.path.join("./Params", 'car.json')
    abalone_params = DatasetParams.create_model_params(dataset_name='contrac', assert_values_flag=False)
    save_dict_to_json(abalone_params, json_path)

if __name__ == '__main__':
    # pass
    gen_param_files()
