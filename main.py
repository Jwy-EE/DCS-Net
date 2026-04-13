import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from data_loader.data_loader import Create_Data_Loader, Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.training import Trainer
from util.utils import fix_seed, log_exp_settings
from util.logger import create_logger
from models.model import DCS_Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='2016.10a')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--target_snrs', type=str, default='all',
                        help='Select specific SNR values')
    parser.add_argument('--lr_scheduler', type=str, default='default',
                        help='Learning rate scheduler type: default or cosine')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--monitor', type=str, default='acc')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training from')
    
    parser.add_argument('--milestone_step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    args = parser.parse_args()

    fix_seed(args.seed)
    
    cfg = Config(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
        num_classes=args.num_classes,
        monitor=args.monitor,
        milestone_step=args.milestone_step,
        gamma=args.gamma
    )
    
    cfg = merge_args2cfg(cfg, vars(args))
    logger = create_logger('AMC-Net')
    log_exp_settings(logger, cfg)

    model = DCS_Net(num_classes=cfg.num_classes).to(args.device)

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=args.device))
        logger.info(f"Loaded pre-trained weights from: {args.resume}")
    elif args.resume:
        logger.warning(f"Weight file not found: {args.resume}")
        logger.info("Starting training from scratch.")


    Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger)

    target_snrs_clean = args.target_snrs.strip()
    if target_snrs_clean.lower() != 'all':
        try:
            target_snr_list = [int(s.strip()) for s in target_snrs_clean.split(',')]

            if isinstance(SNRs, list):
                snr_array = np.array(SNRs)
            else:
                snr_array = SNRs

            mask = np.isin(snr_array, target_snr_list)

            Signals = Signals[mask]
            Labels = Labels[mask]
            SNRs_filtered = []
            for i, keep in enumerate(mask):
                if keep:
                    SNRs_filtered.append(SNRs[i])
            SNRs = SNRs_filtered

            snrs = sorted(list(set(SNRs)))
        except ValueError:
            logger.error(f"Error: target_snrs parameter format is incorrect. Please use the format like '-20,-18,-16'.")
            exit()

    train_set, test_set, val_set, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    Signals_test, Labels_test = test_set

    train_loader, val_loader = Create_Data_Loader(train_set, val_set, cfg, logger)
    trainer = Trainer(model, train_loader, val_loader, cfg, logger)
    trainer.loop()

    logger.info("=== Training Complete, Starting Testing ===")
    if args.target_snrs.strip().lower() != 'all':
        snr_suffix = f"_snr{args.target_snrs.replace(',', '_').replace('-', 'n')}"
    else:
        snr_suffix = ""
    save_model_name = f"{cfg.dataset}_{snr_suffix}.pkl"
    os.makedirs(args.ckpt_path, exist_ok=True)
    best_weight = os.path.join(args.ckpt_path, save_model_name)

    if os.path.exists(best_weight):
        model.load_state_dict(torch.load(best_weight, map_location=args.device))
        logger.info(f"Auto-loaded best training weights: {best_weight}")
    else:
        logger.error(f"Error: Could not locate the saved model file {best_weight}")
        exit()

    y_true_raw = Labels_test
    y_true_all = y_true_raw.argmax(axis=1) if len(y_true_raw.shape) > 1 else y_true_raw
    if isinstance(y_true_all, torch.Tensor):
        y_true_all = y_true_all.cpu().numpy()

    test_snrs = np.array(SNRs)[test_idx]
    unique_snrs = sorted(np.unique(test_snrs))

    logger.info("=== Test Start ===")

    acc_list = []

    model.eval()
    with torch.no_grad():
        for snr in unique_snrs:
            mask = (test_snrs == snr)
            x_snr = Signals_test[mask]
            y_snr = y_true_all[mask]

            preds = []
            for i in range(0, len(x_snr), args.batch_size):
                bx = x_snr[i:i + args.batch_size]
                if not isinstance(bx, torch.Tensor):
                    bx = torch.from_numpy(bx).to(args.device).float()
                else:
                    bx = bx.to(args.device).float()
                preds.append(model(bx).argmax(dim=1).cpu().numpy())

            y_pred = np.concatenate(preds)
            acc = np.mean(y_pred == y_snr) * 100
            acc_list.append(acc)

            logger.info(f"SNR {snr:3d} dB | Acc: {acc:6.2f}%")

    logger.info("-" * 70)
    logger.info(f"Overall Average Accuracy: {np.mean(acc_list):.2f}%")
    logger.info("=" * 70)
