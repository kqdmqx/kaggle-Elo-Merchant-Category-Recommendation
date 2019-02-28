import datetime
import sys
import logging
from filepath_collection import gen_log_filepath

global_logger = None
global_slience_flag = False


def set_global_slience_flag(flag):
    global global_slience_flag
    global_slience_flag = flag


def get_logger(name):
    return logging.getLogger(name)


def init_logger(name, to_console=True, to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')

    if to_console:
        # console handler for validation info
        ch_va = logging.StreamHandler(sys.stdout)
        ch_va.setLevel(logging.INFO)
        ch_va.setFormatter(fmt=message_format)
        logger.addHandler(ch_va)

    if to_file:
        # file handler for validation info
        filepath = gen_log_filepath(name)
        fh_va = logging.FileHandler(filepath)
        fh_va.setFormatter(fmt=message_format)
        logger.addHandler(fh_va)

    return logger


def init_global_logger(name):
    global global_logger
    global_logger = init_logger(name)


def print_info(info, obj, level="info"):
    global global_logger
    global global_slience_flag
    if global_slience_flag:
        return

    info = "[{}] {} : {}".format(level, info, obj)
    if global_logger is not None:
        global_logger.info(info)
    else:
        print(info)


class Timer:

    def __init__(self, silence=False):
        self.silence = silence

    def start(self, info):
        self.start_time = datetime.datetime.now()
        if not self.silence:
            print_info("[timer.start]", self.start_info(info))

    def end(self, info):
        self.end_time = datetime.datetime.now()
        if not self.silence:
            print_info("[timer.end]", self.end_info(info))

    def start_info(self, info):
        return "[{}]".format(info)

    def end_info(self, info):
        return "[{}] cost {}".format(info, self.end_time - self.start_time)


def main():
    pass


if __name__ == '__main__':
    main()
