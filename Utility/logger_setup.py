import logging


def get_logger(module_name):
    formatter = logging.Formatter(
        fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger