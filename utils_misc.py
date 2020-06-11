import torch
import os
import random
import numpy as np
import pickle
import datetime
import os.path


def prepare_save_dir(args, filename):
    """ Create saving directory."""
    runner_name = os.path.basename(filename).split(".")[0]
    model_dir = 'data/experiments/{}/{}/'.format(runner_name, args.name)
    args.savedir = model_dir
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    return args


def write_txt(args, string):
    """ Write the string in a text file."""
    with open(args.savedir + 'out.txt', 'a') as f:
        f.write(string + " \n")


def seed_torch(seed=1029):
    """ Seed the run."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_logger(args, metrics):
    """ Create a logger."""
    args.logger = {}
    args.logger['time_start'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    i = 0
    while os.path.isfile(args.savedir + str(i) + '.pkl'):
        i += 1
    args.logger['pkl_path'] = args.savedir + str(i) + '.pkl'
    args.logger['path'] = args.savedir + str(i)
    for metric in metrics:
        args.logger[metric] = []
    return args


def save_logger(args):
    """ Save the logger."""
    with open(args.logger['pkl_path'], "wb") as output_file:
        pickle.dump(vars(args), output_file)
