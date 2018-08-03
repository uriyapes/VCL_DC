import os
import logging
from datetime import datetime
import json
import numpy as np


def set_a_logger(log_name='log', dirpath="./", filename=None, console_level=logging.DEBUG, file_level=logging.DEBUG):
    """
    Returns a logger object which logs messages to file and prints them to console.
    If you want to log messages from different modules you need to use the same log_name in all modules,  by doing so
    all the modules will print to the same files (created by the first module).
    By default, when using the logger, every new run will generate a new log file - filename_timestamp.log.
    If you wish to write to an existing file you should set the dirpath and filename params to the path of the file and
    make sure you are the first to call set_a_logger with log_name.
    :param log_name: The logger name, use the same name from different modules to write to the same file. In case no filename
                      is given the log_name will used to create the filename (date and .log are added automatically).
    :param dirpath: the logs directory.
    :param filename: the name of the file without any suffix.
    :param console_level: logging level to the console (screen).
    :param file_level: logging level to the file.
    :return: a logger object.
    """

    assert type(log_name) == str
    assert type(dirpath) == str
    assert type(console_level) == int
    assert type(file_level) == int

    if filename:
        assert type(filename) == str
    else:
        timestamp = "_" + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        filename = log_name + timestamp + ".log"
    filepath = os.path.join(dirpath, filename)

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.DEBUG)

    if not logger.handlers:
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(console_level)

        fh = logging.FileHandler(filepath)
        fh.setLevel(file_level)
        # create formatter
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    # 'application' code
    logger.debug('debug message is shown')
    logger.info('info message  is shown')
    logger.warn('warn message is shown')
    logger.error('error message is shown')
    return logger


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



def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


if __name__ == '__main__':
    # Test the logger wrap function - write inside log.log
    logger_name = 'log'
    dirpath = "./Logs"
    logger = set_a_logger(logger_name, dirpath)
    logger.debug('log')

    # Test the logger wrap function - create a different log file and write inside it
    logger_name = 'log2'
    logger2 = set_a_logger(logger_name, dirpath)
    logger2.debug('log2')

    # Test that getting the logger from different module is possible by writing to the same file the logger used.
    logger2_diff_module = set_a_logger(logger_name)
    logger2_diff_module.debug('logger2_diff_module')

    # Test save_dict_to_json function
    json_path = os.path.join("./Logs", 'json')
    d = {'a' : 3, 'b' : np.array([2.3233554])}
    save_dict_to_json(d,  json_path)

    # Test the params class
    params = Params(json_path)
    logger2.critical(params.b)
    params.b = 2
    params.save(json_path)

    params.save(json_path + "2")
