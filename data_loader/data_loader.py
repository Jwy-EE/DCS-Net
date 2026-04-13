import pickle
import torch
import numpy as np
import torch.utils.data as Data

def Load_Dataset(dataset, logger):
    if dataset != '2016.10a':
        raise NotImplementedError(f'Not Implemented dataset:{dataset}')
    
    classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
               b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
    
    file_pointer = r"data\radioml\RML2016.10a_Aerospace_corrupted.pkl"
    
    Signals = []
    Labels = []
    SNRs = []
    
    Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])
    
    for mod in mods:
        for snr in snrs:
            Signals.append(Set[(mod, snr)])
            for i in range(Set[(mod, snr)].shape[0]):
                Labels.append(mod)
                SNRs.append(snr)
    
    Signals = np.vstack(Signals)
    Signals = torch.from_numpy(Signals.astype(np.float32))
    
    Labels = [classes[i] for i in Labels] 
    Labels = np.array(Labels, dtype=np.int64)
    Labels = torch.from_numpy(Labels)
    
    return Signals, Labels, SNRs, snrs, mods

def Dataset_Split(Signals,
                  Labels,
                  snrs,
                  mods,
                  logger,
                  val_size=0.2,
                  test_size=0.2):
    global test_idx
    n_examples = Signals.shape[0]
    n_train = int(n_examples * (1 - val_size - test_size))

    train_idx = []
    test_idx = []
    val_idx = []

    Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

    for k in range(0, Slices_list.shape[0] - 1):
        train_idx_subset = np.random.choice(
            range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
            replace=False)
        Test_Val_idx_subset = list(set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
        test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                           size=int(
                                               (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * (test_size + val_size))),
                                           replace=False)
        val_idx_subset = list(set(Test_Val_idx_subset) - set(test_idx_subset))

        train_idx = np.hstack([train_idx, train_idx_subset])
        val_idx = np.hstack([val_idx, val_idx_subset])
        test_idx = np.hstack([test_idx, test_idx_subset])

    train_idx = train_idx.astype('int64')
    val_idx = val_idx.astype('int64')
    test_idx = test_idx.astype('int64')

    Signals_train = Signals[train_idx]
    Labels_train = Labels[train_idx]

    Signals_test = Signals[test_idx]
    Labels_test = Labels[test_idx]

    Signals_val = Signals[val_idx]
    Labels_val = Labels[val_idx]


    return (Signals_train, Labels_train), \
           (Signals_test, Labels_test), \
           (Signals_val, Labels_val), \
            test_idx


def Create_Data_Loader(train_set, val_set, cfg, logger):

    train_data = Data.TensorDataset(*train_set)
    val_data = Data.TensorDataset(*val_set)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = Data.DataLoader(
        dataset=val_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader
