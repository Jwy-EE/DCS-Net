import logging
import os
import io
import sys

def create_logger(filename, file_handle=True):
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(levelname)s:%(message)s')
    ch.setFormatter(stream_formatter)
    
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                msg = msg.replace('░', '.')
                msg = msg.replace('█', '#')
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    safe_ch = SafeStreamHandler()
    safe_ch.setLevel(logging.INFO)
    safe_ch.setFormatter(stream_formatter)
    logger.addHandler(safe_ch)
    
    if file_handle:
        log_dir = 'log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'log.txt')
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(stream_formatter)
        logger.addHandler(fh)
    
    return logger

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
