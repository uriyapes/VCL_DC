import os
import logging
from datetime import datetime


def set_a_logger(log_name='log', console_level=logging.DEBUG, file_level=logging.DEBUG):
    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level=logging.DEBUG)

    if not logger.handlers:
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(console_level)

        fh = logging.FileHandler(log_name)
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


if __name__ == '__main__':
    filename = os.path.join("./Logs", 'log.log')
    logger = set_a_logger(filename)
    logger.debug('log')
    t = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    filename = os.path.join("./Logs", 'evaluate' + t + '.log')
    logger2 = set_a_logger(filename)
    logger2.debug('log2')

