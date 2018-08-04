import os
import logging
from datetime import datetime


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
    :param filename: if value is specified the name of the file will be filename without any suffix.
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
    logger.critical('Logging level inside file is: {}'.format(logging._levelNames[file_level]))
    return logger


if __name__ == '__main__':
    # Test the logger wrap function - write inside log.log
    logger_name = 'example'
    dirpath = "./Logs"
    logger = set_a_logger(logger_name, dirpath)
    logger.debug('log')

    # Test the logger wrap function - create a different log file and write inside it
    logger_name = 'example2'
    logger2 = set_a_logger(logger_name, dirpath)
    logger2.debug('log2')

    # Test that getting the logger from different module is possible by writing to the same file the logger used.
    logger2_diff_module = set_a_logger(logger_name)
    logger2_diff_module.debug('example2_diff_module')

