import os
import json
import numpy as np


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


class ModelParams(Params):
    def __init__(self, json_path):
        super(ModelParams, self).__init__(json_path)

    @classmethod
    def _create_default_model_params(cls):
        dict = {}
        dict['batch norm'] = 0
        dict['keep prob for dropout'] = 0.5
        dict['number of layers'] = 4
        dict['activation'] = 'RELU'
        dict['random seeds'] = 1
        dict['tf seed'] = 230
        dict['np seed'] = 100
        dict['number of epochs'] = 500
        return dict






def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: float(v) for k, v in d.items()}
        for k,v in d.items():
            if type(v) != str:
                v = float(v)
            d[k] = v
        json.dump(d, f, indent=4)

if __name__ == '__main__':
    # Test save_dict_to_json function
    json_path = os.path.join("./Params", 'json')
    d = {'a': 3, 'b': np.array([2.3233554]), 'c': 'hello'}
    save_dict_to_json(d, json_path)

    # Test the params class
    params = Params(json_path)
    params.b = 2
    params.save(json_path)

    params.save(json_path + "2")

    # Create template for model params json file
    json_path = os.path.join("./Params", 'model_params_template.json')
    dict_params = ModelParams._create_default_model_params()
    save_dict_to_json(dict_params, json_path)
    model_params = ModelParams(json_path)
    print model_params.dict

    model_params.dict['number of layers'] = 8
    json_path = os.path.join("./Params", 'depth_8.json')
    model_params.save(json_path)
