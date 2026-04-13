import os.path

def merge_args2cfg(cfg, args_dict):
    for k, v in args_dict.items():
        setattr(cfg, k, v)
    return cfg

class Config:
    def __init__(self, dataset, 
                 epochs=100, batch_size=256, patience=20, lr=0.001,
                 num_classes=11, monitor='acc',
                 milestone_step=3, gamma=0.1):
        self.dataset = dataset

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr

        self.num_classes = num_classes
        self.monitor = monitor
        
        self.milestone_step = milestone_step
        self.gamma = gamma
        self.test_batch_size = batch_size 

        if self.dataset == '2016.10a':
            self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                            b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
        else:
            raise NotImplementedError(f'Not Implement dataset:{self.dataset}')
