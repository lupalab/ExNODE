import os
import json
import time
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
import traceback
import pickle

from utils import makedirs
from classifier import ClassificationEngine
from point_cloud import get_loader

SOLVER = ['euler', 'midpoint', 'rk4', 'dopri5']
DSET = ['spacial_mnist', 'modelnet']
MODEL = {'classification': ClassificationEngine}

def parse_command():
    parser = argparse.ArgumentParser('SetODE')
    parser.add_argument('--model', type=str, default='flow', choices=['flow', 'vae', 't_series', 'classification'], help='which model to run')
    parser.add_argument('--sub_model', type=str, default='transformer', choices=['odedset', 'odetrans', 'deepset'], help='which sub model to use')
    # dataset settings
    parser.add_argument('--dset', type=str, default='spacial_mnist', choices=DSET, help='which dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--set_size', type=int, default=50, help='number of points to use')
    parser.add_argument('--pts_dim', type=int, default=3, help='dim of point cloud')
    parser.add_argument('--categories', type=str, default='', help='which shape to use')
    parser.add_argument('--aug_data', type=eval, default=False, help='whether to augment training data.')
    # method settings
    parser.add_argument('--dims', type=str, default='64,256', help='dimension used in diffeq')
    parser.add_argument('--fc_dims', type=str, default='128', help='dimension of fc layers')
    parser.add_argument('--set_hdims', type=str, default='512,512', help='hidden dim in set transformer')
    
    parser.add_argument('--T_end', type=float, default=1.0, help='the end time of neural ode')
    parser.add_argument('--steps', type=int, default=2, help='steps of numerical ode')
    parser.add_argument('--num_blocks', type=int, default=1, help='number of blocks used in model')
    parser.add_argument('--solver', type=str, default='euler', choices=SOLVER, help='numerical solver of ode')
    parser.add_argument('--tol', type=float, default=1e-5, help='the tolerance of numerical ode')
    parser.add_argument('--batch_norm', type=bool, default=True, help='whether to use batch norm')
    # training settings
    parser.add_argument('--epochs', type=int, default=60, help='epoches to train the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate while training the model')
    parser.add_argument('--optimizer', type=str, default='Adam', help='the optimizer to use')
    parser.add_argument('--beta1', type=float, default=0.9, help='deta1 in Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='deta2 in Adam optimizer')
    parser.add_argument('--data_dir', type=str, default='./data', help='the path to the data directory')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size')
    # backend settings
    parser.add_argument('--save', type=str, default='./exp', help='the directory to save log and model')
    parser.add_argument('--gpu', type=int, default=0, help='indicate the gpu to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed while training')
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' # set device to use

    return args


def get_logger(log_path, displaying=True, saving=True, debug=False):
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)

    # setting log file
    if saving:
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    # setting log terminal
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    return logger


def main():
    args = parse_command()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # enable cudnn benchmark to accelerate training
    torch.backends.cudnn.benchmark = True

    # set logging file
    makedirs(args.save)
    logger = get_logger(os.path.join(args.save, 'log'))
    logger.info(json.dumps(vars(args), indent=4))

    # parse args
    args.dims = [int(d) for d in args.dims.split(',')]
    args.fc_dims = [int(d) for d in args.fc_dims.split(',')]
    args.set_hdims = [int(d) for d in args.set_hdims.split(',')]

    # fetch data loader
    train_loader, test_loader = get_loader(args)

    try:
        model = MODEL[args.model](args, logger) # initialize model
        model.set_data(train_loader, test_loader)
        logger.info(model.model) # print model structure
        logger.info(f'Number of parameters: {model.count_parameters()}') # count parameters
        metric_log = model.fit() # training and return the log
    except:
        logging.error(traceback.format_exc())
    
    # pickle and write the metric_log
    with open(os.path.join(args.save, 'log.pkl'), 'wb') as handle:
            pickle.dump(metric_log, handle)

    return 0


if __name__ == '__main__':
    main()
