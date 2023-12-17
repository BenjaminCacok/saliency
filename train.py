import logging
import os
import random
import time
from datetime import datetime

import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

from config import cfg
from engine_train import train_one_epoch_salicon, validation_one_epoch_salicon
from loss import *
from models.models import GSGNet_T
from utils import SaliconT, SaliconVal


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Output to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output to terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_type = "type3_"
experiment_dir = experiment_type + experiment_time


def set_seeds(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    os.makedirs(f"./logs/{experiment_dir}", exist_ok=True)
    # Load Dataset
    train_pd = pd.read_csv(cfg.DATA.SALICON_TRAIN)
    val_pd = pd.read_csv(cfg.DATA.SALICON_VAL)
    trainset = SaliconT(cfg.DATA.SALICON_ROOT, train_pd['X'], train_pd['Y'], size=cfg.DATA.RESOLUTION)
    valset = SaliconVal(cfg.DATA.SALICON_ROOT, val_pd['X'], val_pd['Y'], size=cfg.DATA.RESOLUTION)

    train_loader = DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, pin_memory=False)
    val_loader = DataLoader(valset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=False)

    total_batch_size = cfg.TRAIN.BATCH_SIZE
    num_training_steps_per_epoch = len(trainset) // total_batch_size
    num_testing_steps_per_epoch = len(valset) // total_batch_size

    # Initialize model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GSGNet_T().to(device)
    optimizer = opt.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    lr_schedule_values_by_epoch = []
    _LR = cfg.SOLVER.LR
    for i in range(cfg.SOLVER.MAX_EPOCH):
        lr_schedule_values_by_epoch.append(_LR)
        if i in {1, 6, 11}:
            _LR = _LR * 0.01
        _LR = max(_LR, cfg.SOLVER.MIN_LR)

    logger = get_logger(f'./logs/{experiment_dir}/train.log')
    logger.info('start training!')

    for epoch in range(cfg.SOLVER.MAX_EPOCH):

        gc.collect()
        torch.cuda.empty_cache()

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values_by_epoch[epoch]

        training_loss = train_one_epoch_salicon(model, optimizer, train_loader, device=device, epoch=epoch,
                                                num_training_steps_per_epoch=num_training_steps_per_epoch)

        val_kl, val_cc, val_nss = validation_one_epoch_salicon(model, val_loader, device=device, epoch=epoch,
                                                               num_testing_steps_per_epoch=num_testing_steps_per_epoch)

        logger.info(
            'Epoch:[{}/{}]\t val_kl={:.4f}\t val_cc={:.4f}\t val_nss={:.4f}\t training_loss = {:.4f}'.format(epoch + 1,
                                                                                                             cfg.SOLVER.MAX_EPOCH,
                                                                                                             val_kl,
                                                                                                             val_cc,
                                                                                                             val_nss,
                                                                                                             training_loss))

        if epoch > 25:
            torch.save(model, os.path.join(f"./logs/{experiment_dir}", "ep{}_.pt".format(epoch + 1)))

    logger.info('finish training!')


if __name__ == "__main__":
    set_seeds()
    main()
